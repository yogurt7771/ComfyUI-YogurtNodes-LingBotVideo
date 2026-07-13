import importlib
import json
import sys
import types

import pytest


torch = pytest.importorskip("torch")


def _load_modules():
    for name in list(sys.modules):
        if name == "yogurt_lingbot_video" or name.startswith("yogurt_lingbot_video."):
            sys.modules.pop(name)
    package = importlib.import_module("yogurt_lingbot_video")
    runtime = importlib.import_module("yogurt_lingbot_video.runtime")
    return package, runtime


def _rewriter_node(package):
    return next(
        node_class()
        for node_class in package.NODE_CLASS_MAPPINGS.values()
        if node_class.RETURN_NAMES == ("prompt_json", "detailed_prompt")
    )


class _Model:
    def __init__(self, events):
        self.events = events

    def to(self, device):
        self.events.append(("to", str(device)))
        return self


class _Backend:
    def __init__(self, events, fail_on=None):
        self.events = events
        self.fail_on = fail_on
        self.model = _Model(events)
        self.call_count = 0

    def generate(self, text, image, use_lora):
        self.call_count += 1
        self.events.append(("generate", self.call_count, use_lora))
        if self.call_count == self.fail_on:
            raise RuntimeError("generation failed")
        if use_lora:
            return json.dumps({"caption": "Mapped caption", "duration": 5})
        return "Detailed expanded prompt"

    def release_vram(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()


class _Handle:
    def __init__(self, backend):
        self.backend = backend


def _cuda_cache_spy(monkeypatch, runtime, events):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty_cache",)))


def test_rewrite_node_exposes_release_boolean_defaulting_true(folder_paths_stub):
    package, _ = _load_modules()
    inputs = _rewriter_node(package).INPUT_TYPES()
    specifications = {**inputs.get("required", {}), **inputs.get("optional", {})}

    release = specifications["release_rewriter_after_rewrite"]

    assert release[0] == "BOOLEAN"
    assert release[1]["default"] is True


def test_release_enabled_preserves_outputs_and_releases_after_both_generations(
    folder_paths_stub, monkeypatch
):
    package, runtime = _load_modules()
    events = []
    backend = _Backend(events)
    _cuda_cache_spy(monkeypatch, runtime, events)

    result = _rewriter_node(package).rewrite(
        _Handle(backend), "Short prompt", "t2v", 5.0,
        release_rewriter_after_rewrite=True,
    )

    assert json.loads(result[0]) == {"caption": "Mapped caption", "duration": 5}
    assert result[1] == "Detailed expanded prompt"
    assert events == [
        ("generate", 1, False),
        ("generate", 2, True),
        ("to", "cpu"),
        ("empty_cache",),
    ]


def test_release_disabled_keeps_current_residency_and_does_not_clear_cache(
    folder_paths_stub, monkeypatch
):
    package, runtime = _load_modules()
    events = []
    backend = _Backend(events)
    _cuda_cache_spy(monkeypatch, runtime, events)

    result = _rewriter_node(package).rewrite(
        _Handle(backend), "Short prompt", "t2v", 5.0,
        release_rewriter_after_rewrite=False,
    )

    assert json.loads(result[0])["caption"] == "Mapped caption"
    assert result[1] == "Detailed expanded prompt"
    assert events == [("generate", 1, False), ("generate", 2, True)]


@pytest.mark.parametrize("fail_on", [1, 2])
def test_release_enabled_releases_when_either_generation_raises(
    folder_paths_stub, monkeypatch, fail_on
):
    package, runtime = _load_modules()
    events = []
    backend = _Backend(events, fail_on=fail_on)
    _cuda_cache_spy(monkeypatch, runtime, events)

    with pytest.raises(RuntimeError, match="generation failed"):
        _rewriter_node(package).rewrite(
            _Handle(backend), "Short prompt", "t2v", 5.0,
            release_rewriter_after_rewrite=True,
        )

    assert events[-2:] == [("to", "cpu"), ("empty_cache",)]
    assert sum(event[0] == "generate" for event in events) == fail_on


class _Inputs(dict):
    def to(self, device):
        return self


class _ReloadProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return "chat"

    def __call__(self, *args, **kwargs):
        return _Inputs(input_ids=torch.zeros((1, 1), dtype=torch.long))

    def batch_decode(self, *args, **kwargs):
        return ["reloaded result"]


class _ReloadModel:
    device = "cpu"

    def __init__(self, identity):
        self.identity = identity
        self.generate_calls = 0

    def disable_adapter(self):
        from contextlib import nullcontext

        return nullcontext()

    def generate(self, **kwargs):
        self.generate_calls += 1
        return torch.zeros((1, 2), dtype=torch.long)

    def to(self, device):
        return self


def test_released_cached_backend_reloads_model_before_next_generate(
    folder_paths_stub, monkeypatch
):
    _, runtime = _load_modules()
    original = _ReloadModel("original")
    replacement = _ReloadModel("replacement")
    loads = []

    def model_loader():
        loads.append("load")
        return replacement

    backend = runtime._TransformersRewriterBackend(
        _ReloadProcessor(), original, model_loader=model_loader
    )
    accelerate = types.ModuleType("accelerate")
    hooks = types.ModuleType("accelerate.hooks")
    hooks.remove_hook_from_submodules = lambda model: None
    accelerate.hooks = hooks
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.hooks", hooks)
    comfy = importlib.import_module("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.soft_empty_cache = lambda: None
    monkeypatch.setattr(comfy, "model_management", model_management, raising=False)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    backend.release_vram()
    result = backend.generate("prompt", None, use_lora=True)

    assert result == "reloaded result"
    assert loads == ["load"]
    assert original.generate_calls == 0
    assert replacement.generate_calls == 1


class _GenerateAndReleaseFailBackend:
    def __init__(self):
        self.release_calls = 0

    def generate(self, *args, **kwargs):
        raise RuntimeError("original generate failure")

    def release_vram(self):
        self.release_calls += 1
        raise RuntimeError("secondary release failure")


def test_generate_failure_is_not_replaced_by_release_failure(folder_paths_stub):
    _, runtime = _load_modules()
    backend = _GenerateAndReleaseFailBackend()

    with pytest.raises(RuntimeError, match="original generate failure"):
        runtime.rewrite_prompt(
            _Handle(backend), "prompt", "t2v", 5.0,
            release_rewriter_after_rewrite=True,
        )

    assert backend.release_calls == 1


def test_ti2v_validation_failure_before_generation_does_not_release(folder_paths_stub):
    _, runtime = _load_modules()
    events = []
    backend = _Backend(events)

    with pytest.raises(ValueError, match="IMAGE"):
        runtime.rewrite_prompt(
            _Handle(backend), "prompt", "ti2v", 5.0,
            release_rewriter_after_rewrite=True,
        )

    assert events == []
