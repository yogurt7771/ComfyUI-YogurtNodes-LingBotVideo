import importlib
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def _load_plugin_root(repo_root):
    for name in list(sys.modules):
        if name == "yogurt_lingbot_video" or name.startswith("yogurt_lingbot_video."):
            sys.modules.pop(name)
    return importlib.import_module("yogurt_lingbot_video")


def _all_inputs(node_class):
    declared = node_class.INPUT_TYPES()
    return {**declared.get("required", {}), **declared.get("optional", {})}


def _default_arguments(node_class):
    arguments = {}
    for name, spec in _all_inputs(node_class).items():
        if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            arguments[name] = spec[1]["default"]
    return arguments


def _execute(node, arguments):
    return getattr(node, node.FUNCTION)(**arguments)


def _generator(plugin):
    node_class = next(node for node in plugin.NODE_CLASS_MAPPINGS.values() if node.RETURN_TYPES == ("IMAGE",))
    return node_class()


def _rewriter(plugin):
    node_class = next(
        node
        for node in plugin.NODE_CLASS_MAPPINGS.values()
        if "prompt" in _all_inputs(node) and "STRING" in node.RETURN_TYPES and node.RETURN_TYPES != ("IMAGE",)
    )
    return node_class()


class _StubVideoPipeline:
    def __init__(self, result=None, result_factory=None):
        self.result = result
        self.result_factory = result_factory
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        frames = self.result_factory(kwargs) if self.result_factory else self.result
        return SimpleNamespace(frames=frames)

    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)

    def run_inference(self, *args, **kwargs):
        return self(*args, **kwargs)


class _StubVideoHandle:
    def __init__(self, pipeline, mode="t2v"):
        self.pipeline = pipeline
        self.pipe = pipeline
        self.device = torch.device("cpu")
        self.mode = mode
        self.moe_sequential_cpu_offload = False
        self.moe_block_swap = False
        self.task = mode
        pipeline._execution_device = self.device
        pipeline.mode = mode
        pipeline.task = mode

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def run_inference(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)


def _generation_arguments(node, model, **overrides):
    inputs = _all_inputs(type(node))
    model_key = next(
        name
        for name, spec in inputs.items()
        if isinstance(spec[0], str) and ("MODEL" in spec[0] or "PIPELINE" in spec[0])
    )
    image_key = next((name for name, spec in inputs.items() if spec[0] == "IMAGE"), None)
    mode = overrides.pop("mode", "t2v")
    image_provided = "image" in overrides
    image = overrides.pop("image", None)
    model.mode = mode
    model.task = mode
    model.pipeline.mode = mode
    model.pipeline.task = mode
    arguments = _default_arguments(type(node))
    arguments.update(
        {
            model_key: model,
            "prompt": "A robot carefully picks up a cup.",
            "width": 32,
            "height": 32,
            "num_frames": 5,
        }
    )
    if image_provided:
        assert image_key is not None
        arguments[image_key] = image
    arguments.update(overrides)
    return arguments


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 30},
        {"height": 31},
        {"num_frames": 0},
        {"num_frames": 2},
        {"num_frames": 6},
    ],
)
def test_generation_rejects_invalid_dimensions_and_frame_counts_before_pipeline(
    repo_root, folder_paths_stub, overrides
):
    plugin = _load_plugin_root(repo_root)
    pipeline = _StubVideoPipeline(result=np.zeros((1, 1, 32, 32, 3), dtype=np.float32))
    node = _generator(plugin)

    with pytest.raises((ValueError, RuntimeError), match="(?i)(width|height|multiple|frame|4n|4.*1)"):
        _execute(node, _generation_arguments(node, _StubVideoHandle(pipeline), **overrides))

    assert pipeline.calls == []


def test_ti2v_requires_image_before_pipeline(repo_root, folder_paths_stub):
    plugin = _load_plugin_root(repo_root)
    pipeline = _StubVideoPipeline(result=np.zeros((1, 5, 32, 32, 3), dtype=np.float32))
    node = _generator(plugin)

    with pytest.raises((ValueError, RuntimeError), match="(?i)(image|ti2v|first frame)"):
        _execute(
            node,
            _generation_arguments(
                node,
                _StubVideoHandle(pipeline),
                mode="ti2v",
                image=None,
            ),
        )

    assert pipeline.calls == []


@pytest.mark.parametrize(
    "frames,expected_count",
    [
        (np.full((1, 2, 4, 6, 3), 255, dtype=np.uint8), 2),
        (torch.full((1, 3, 2, 4, 6), 255, dtype=torch.uint8), 2),
        ([np.zeros((4, 6, 3), dtype=np.uint8), np.full((4, 6, 3), 255, dtype=np.uint8)], 2),
    ],
)
def test_generation_normalizes_common_pipeline_frame_results_to_comfy_image(
    repo_root, folder_paths_stub, frames, expected_count
):
    plugin = _load_plugin_root(repo_root)
    pipeline = _StubVideoPipeline(result=frames)
    node = _generator(plugin)

    result = _execute(node, _generation_arguments(node, _StubVideoHandle(pipeline)))
    image = result[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (expected_count, 4, 6, 3)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0


def test_t2i_emits_one_frame_and_accepts_prompt_json_text(repo_root, folder_paths_stub):
    plugin = _load_plugin_root(repo_root)

    def result_factory(kwargs):
        count = kwargs.get("num_frames", kwargs.get("frames", 1))
        return np.zeros((1, count, 4, 6, 3), dtype=np.float32)

    pipeline = _StubVideoPipeline(result_factory=result_factory)
    node = _generator(plugin)
    prompt_json = json.dumps({"caption": "A robot picks up a cup.", "duration": 1})

    result = _execute(
        node,
        _generation_arguments(
            node,
            _StubVideoHandle(pipeline),
            mode="t2i",
            prompt=prompt_json,
            num_frames=5,
        ),
    )

    assert result[0].shape == (1, 4, 6, 3)
    assert len(pipeline.calls) == 1


class _StubRewriterHandle:
    def __init__(self, role, call_order):
        self.role = role
        self.call_order = call_order
        self.calls = []
        self.pipeline = self
        self.model = self

    def _result(self):
        if self.role == "base":
            return "A detailed scene of a robot carefully picking up a cup."
        return json.dumps({"caption": "A robot carefully picks up a cup.", "duration": 2})

    def generate(self, *args, **kwargs):
        self.call_order.append(self.role)
        self.calls.append((args, kwargs))
        return self._result()

    def rewrite(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        yield


class _StubCombinedRewriterHandle:
    def __init__(self, base, adapter):
        self.base = base
        self.adapter = adapter
        self.backend = self
        self.calls = []
        self.release_calls = 0

    def generate(self, text, image=None, use_lora=False, **kwargs):
        self.calls.append((text, image, use_lora))
        stage = self.adapter if use_lora else self.base
        return stage.generate(text, image=image, use_lora=use_lora)

    def release_vram(self):
        self.release_calls += 1


def _rewriter_handle_key(node):
    inputs = _all_inputs(type(node))
    return next(
        (
            name
            for name, spec in inputs.items()
            if (isinstance(spec[0], str) and "REWRITER" in spec[0])
            or "rewriter" in name.lower()
        ),
        None,
    )


def _rewriter_arguments(node, rewriter, **overrides):
    inputs = _all_inputs(type(node))
    handle_key = _rewriter_handle_key(node)
    assert handle_key is not None
    mode_key = next(
        name
        for name, spec in inputs.items()
        if isinstance(spec[0], (tuple, list)) and set(spec[0]) == {"t2i", "t2v", "ti2v"}
    )
    image_key = next((name for name, spec in inputs.items() if spec[0] == "IMAGE"), None)
    mode = overrides.pop("mode", "t2v")
    image_provided = "image" in overrides
    image = overrides.pop("image", None)
    arguments = _default_arguments(type(node))
    arguments.update(
        {
            handle_key: rewriter,
            mode_key: mode,
            "prompt": "A robot picks up a cup.",
        }
    )
    if image_provided:
        assert image_key is not None
        arguments[image_key] = image
    arguments.update(overrides)
    return arguments


def test_rewriter_accepts_combined_base_and_adapter_handle(repo_root, folder_paths_stub):
    plugin = _load_plugin_root(repo_root)
    node = _rewriter(plugin)

    assert _rewriter_handle_key(node) is not None


def test_rewriter_ti2v_requires_image_before_model_calls(repo_root, folder_paths_stub):
    plugin = _load_plugin_root(repo_root)
    node = _rewriter(plugin)
    call_order = []
    base = _StubRewriterHandle("base", call_order)
    adapter = _StubRewriterHandle("adapter", call_order)
    rewriter = _StubCombinedRewriterHandle(base, adapter)

    with pytest.raises((ValueError, RuntimeError), match="(?i)(image|ti2v|first frame)"):
        _execute(
            node,
            _rewriter_arguments(node, rewriter, mode="ti2v", image=None),
        )

    assert call_order == []
    assert rewriter.calls == []


def test_rewriter_runs_expand_then_json_mapping_and_returns_both_texts(repo_root, folder_paths_stub):
    plugin = _load_plugin_root(repo_root)
    node = _rewriter(plugin)
    call_order = []
    base = _StubRewriterHandle("base", call_order)
    adapter = _StubRewriterHandle("adapter", call_order)
    rewriter = _StubCombinedRewriterHandle(base, adapter)

    result = _execute(node, _rewriter_arguments(node, rewriter))

    assert call_order == ["base", "adapter"]
    assert len(rewriter.calls) == 2
    assert rewriter.release_calls == 1
    assert len(result) == 2
    structured = next(text for text in result if text.lstrip().startswith("{"))
    detailed = next(text for text in result if not text.lstrip().startswith("{"))
    assert json.loads(structured)["caption"] == "A robot carefully picks up a cup."
    assert "detailed scene" in detailed.lower()
