import ast
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_PATH = PLUGIN_ROOT / "yogurt_lingbot_video" / "upstream" / "transformer_lingbot_video.py"
DEFAULTS_PATH = PLUGIN_ROOT / "yogurt_lingbot_video" / "upstream" / "defaults.py"


def _load_attention_class(optimized, optimized_masked):
    source = TRANSFORMER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"LingBotVideoRMSNorm", "apply_rotary_emb", "LingBotVideoAttention"}
    body = [
        node
        for node in tree.body
        if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted)
    ]
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "os": os,
        "Optional": Optional,
        "optimized_attention": optimized,
        "optimized_attention_masked": optimized_masked,
        "flash_attn_varlen_func_v3": None,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(TRANSFORMER_PATH), "exec"), namespace)
    return namespace["LingBotVideoAttention"]


def _official_image_negative_prompt():
    tree = ast.parse(DEFAULTS_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_NEGATIVE_PROMPT_IMAGE"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("DEFAULT_NEGATIVE_PROMPT_IMAGE is missing")


@pytest.mark.parametrize("use_mask", [False, True])
def test_comfy_attention_adapter_matches_official_bshd_contract(monkeypatch, use_mask):
    monkeypatch.setenv("LINGBOT_ATTENTION_BACKEND", "comfy")
    def comfy_attention(q, k, v, heads, mask=None):
        batch, sequence, width = q.shape
        head_dim = width // heads
        q = q.view(batch, sequence, heads, head_dim).transpose(1, 2)
        k = k.view(batch, sequence, heads, head_dim).transpose(1, 2)
        v = v.view(batch, sequence, heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(batch, sequence, width)

    attention_class = _load_attention_class(comfy_attention, comfy_attention)
    torch.manual_seed(1234)
    module = attention_class(8, 2, 1e-6, True, True).eval()
    x = torch.randn(2, 5, 8)
    rotary = torch.polar(torch.ones(2, 5, 2), torch.randn(2, 5, 2))
    mask = None
    if use_mask:
        mask = torch.tensor(
            [[[[True, True, True, False, False]]], [[[True, True, True, True, False]]]]
        )

    actual = module(x, rotary, mask)
    q = module.to_q(x).unflatten(2, (module.num_heads, module.head_dim))
    k = module.to_k(x).unflatten(2, (module.num_heads, module.head_dim))
    v = module.to_v(x).unflatten(2, (module.num_heads, module.head_dim))
    q = torch.view_as_real(
        torch.view_as_complex(module.norm_q(q).float().reshape(2, 5, 2, -1, 2))
        * rotary.unsqueeze(2)
    ).flatten(3).type_as(q)
    k = torch.view_as_real(
        torch.view_as_complex(module.norm_k(k).float().reshape(2, 5, 2, -1, 2))
        * rotary.unsqueeze(2)
    ).flatten(3).type_as(k)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)
    reference = module.to_out(reference.flatten(2, 3).type_as(x))

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_dense_t2i_blank_negative_prompt_uses_official_image_default(repo_root, monkeypatch):
    import sys

    folder_paths = ModuleType("folder_paths")
    folder_paths.models_dir = str(repo_root / "models")
    folder_paths.folder_names_and_paths = {}
    folder_paths.add_model_folder_path = lambda *args, **kwargs: None
    folder_paths.get_folder_paths = lambda name: []
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    sys.path.insert(0, str(repo_root))
    try:
        from yogurt_lingbot_video.handles import LingBotVideoPipelineHandle
        from yogurt_lingbot_video.runtime import generate_frames

        calls = []

        class Pipeline:
            _execution_device = "cpu"

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(images=torch.rand(1, 480, 832, 3))

        handle = LingBotVideoPipelineHandle(Pipeline(), "local", "t2i", "cpu", "float32", False)
        generate_frames(
            handle,
            prompt=("A cinematic mountain observatory at blue hour, with warm window light, " * 8),
            negative_prompt="",
            width=832,
            height=480,
            num_frames=81,
            steps=40,
            guidance_scale=3.0,
            shift=3.0,
            seed=42,
        )

        assert calls[0]["negative_prompt"] == _official_image_negative_prompt()
    finally:
        sys.path.remove(str(repo_root))
