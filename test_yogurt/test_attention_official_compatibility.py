import ast
import os
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

TRANSFORMER_PATH = (
    Path(__file__).resolve().parents[1]
    / "yogurt_lingbot_video"
    / "upstream"
    / "transformer_lingbot_video.py"
)


def _load_attention_class(optimized, optimized_masked, dispatch_attention=None):
    tree = ast.parse(TRANSFORMER_PATH.read_text(encoding="utf-8"))
    wanted = {"LingBotVideoRMSNorm", "apply_rotary_emb", "LingBotVideoAttention"}
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "os": os,
        "Optional": Optional,
        "optimized_attention": optimized,
        "optimized_attention_masked": optimized_masked,
        "dispatch_attention_fn": dispatch_attention,
        "flash_attn_varlen_func_v3": None,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(TRANSFORMER_PATH), "exec"),
        namespace,
    )
    return namespace["LingBotVideoAttention"]


def _official_sdpa_output(module, x, rotary, mask):
    batch, sequence, _ = x.shape
    q = module.to_q(x).unflatten(2, (module.num_heads, module.head_dim))
    k = module.to_k(x).unflatten(2, (module.num_heads, module.head_dim))
    v = module.to_v(x).unflatten(2, (module.num_heads, module.head_dim))
    q = torch.view_as_real(
        torch.view_as_complex(module.norm_q(q).float().reshape(batch, sequence, module.num_heads, -1, 2))
        * rotary.unsqueeze(2)
    ).flatten(3).type_as(q)
    k = torch.view_as_real(
        torch.view_as_complex(module.norm_k(k).float().reshape(batch, sequence, module.num_heads, -1, 2))
        * rotary.unsqueeze(2)
    ).flatten(3).type_as(k)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)
    return module.to_out(out.flatten(2, 3).type_as(x))


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("use_parallel_config", [False, True])
def test_default_attention_uses_official_dispatch_contract(monkeypatch, use_mask, use_parallel_config):
    calls = []

    def official_dispatch(q, k, v, *, attn_mask, parallel_config):
        calls.append((q, k, v, attn_mask, parallel_config))
        return v

    def unexpected_comfy_dispatch(*args, **kwargs):
        raise AssertionError("default attention dispatched to a Comfy attention backend")

    monkeypatch.delenv("LINGBOT_ATTENTION_BACKEND", raising=False)
    attention_class = _load_attention_class(
        unexpected_comfy_dispatch,
        unexpected_comfy_dispatch,
        official_dispatch,
    )
    torch.manual_seed(20260711)
    module = attention_class(128, 2, 1e-6, True, True).eval()
    x = torch.randn(2, 17, 128)
    angles = torch.randn(2, 17, 32)
    rotary = torch.polar(torch.ones_like(angles), angles)
    mask = None
    if use_mask:
        mask = torch.ones(2, 1, 1, 17, dtype=torch.bool)
        mask[0, ..., 11:] = False
        mask[1, ..., 15:] = False

    parallel_config = object() if use_parallel_config else None
    actual = module(x, rotary, mask, parallel_config=parallel_config)

    assert len(calls) == 1
    q, k, v, actual_mask, actual_parallel_config = calls[0]
    assert q.shape == k.shape == v.shape == (2, 17, 2, 64)
    assert actual_mask is mask
    assert actual_parallel_config is parallel_config
    expected = module.to_out(v.flatten(2, 3).type_as(x))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_mask", [False, True])
def test_comfy_attention_override_retains_selected_backend(monkeypatch, use_mask):
    calls = []

    def comfy_attention(q, k, v, heads, mask=None):
        calls.append(mask)
        batch, sequence, width = q.shape
        head_dim = width // heads
        q = q.view(batch, sequence, heads, head_dim).transpose(1, 2)
        k = k.view(batch, sequence, heads, head_dim).transpose(1, 2)
        v = v.view(batch, sequence, heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(batch, sequence, width)

    monkeypatch.setenv("LINGBOT_ATTENTION_BACKEND", "comfy")
    def unexpected_official_dispatch(*args, **kwargs):
        raise AssertionError("Comfy override dispatched to the official attention backend")

    attention_class = _load_attention_class(
        comfy_attention,
        comfy_attention,
        unexpected_official_dispatch,
    )
    torch.manual_seed(20260711)
    module = attention_class(128, 2, 1e-6, True, True).eval()
    x = torch.randn(2, 17, 128)
    angles = torch.randn(2, 17, 32)
    rotary = torch.polar(torch.ones_like(angles), angles)
    mask = None
    if use_mask:
        mask = torch.ones(2, 1, 1, 17, dtype=torch.bool)
        mask[0, ..., 11:] = False
        mask[1, ..., 15:] = False

    actual = module(x, rotary, mask)
    expected = _official_sdpa_output(module, x, rotary, mask)

    assert calls == [mask]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
