import sys
import types
import json
import importlib
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a modern living room", {"comprehensive_description": "a modern living room"}),
        (json.dumps({"caption": {"shot": "wide", "subject": "room"}, "duration": 5}), {"shot": "wide", "subject": "room"}),
        (json.dumps({"shot": "wide", "duration": 5, "fps": 24, "width": 832}), {"shot": "wide"}),
        (json.dumps([{"shot": "wide"}, {"shot": "close"}]), [{"shot": "wide"}, {"shot": "close"}]),
    ],
)
def test_generate_frames_normalizes_prompt_for_official_dit(fresh_import, prompt, expected):
    runtime = fresh_import("yogurt_lingbot_video.runtime")

    class Pipeline:
        _execution_device = torch.device("cpu")

        def __init__(self):
            self.prompt = None

        def __call__(self, **kwargs):
            self.prompt = kwargs["prompt"]
            return SimpleNamespace(frames=np.zeros((1, 1, 4, 4, 3), dtype=np.float32))

    pipeline = Pipeline()
    handle = SimpleNamespace(mode="t2v", device=torch.device("cpu"), pipeline=pipeline, moe_sequential_cpu_offload=False, moe_block_swap=False)
    runtime.generate_frames(handle, prompt, "negative", 16, 16, 5, 1, 1.0, 3.0, 1)

    assert json.loads(pipeline.prompt) == expected
    assert pipeline.prompt == json.dumps(expected, ensure_ascii=False, separators=(",", ":"))


def test_generate_frames_rejects_empty_prompt_before_pipeline(fresh_import):
    runtime = fresh_import("yogurt_lingbot_video.runtime")

    class Pipeline:
        _execution_device = torch.device("cpu")

        def __call__(self, **kwargs):
            raise AssertionError("pipeline must not run for an empty prompt")

    handle = SimpleNamespace(mode="t2v", device=torch.device("cpu"), pipeline=Pipeline(), moe_sequential_cpu_offload=False, moe_block_swap=False)
    with pytest.raises(ValueError, match="(?i)prompt.*empty|empty.*prompt"):
        runtime.generate_frames(handle, "  ", "negative", 16, 16, 5, 1, 1.0, 3.0, 1)


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.empty(0))

    def forward(self, latent, timestep, prompt, **kwargs):
        return (torch.zeros_like(latent),)


class _Scheduler:
    sigma_max = 1.0
    sigma_min = 0.0

    def set_timesteps(self, steps, **kwargs):
        self.timesteps = torch.arange(steps, 0, -1)

    def step(self, noise, timestep, latents, **kwargs):
        return (latents + 1,)


def _load_pipeline_types(monkeypatch):
    class DiffusionPipeline(torch.nn.Module):
        pass

    diffusers = types.ModuleType("diffusers")
    diffusers.DiffusionPipeline = DiffusionPipeline
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.BaseOutput = object
    torch_utils = types.ModuleType("diffusers.utils.torch_utils")
    torch_utils.randn_tensor = torch.randn
    scheduler = types.ModuleType("yogurt_lingbot_video.upstream.scheduling_flow_unipc")
    scheduler.FlowUniPCMultistepScheduler = type("FlowUniPCMultistepScheduler", (), {})
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)
    monkeypatch.setitem(sys.modules, "diffusers.utils.torch_utils", torch_utils)
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    for name in (
        "yogurt_lingbot_video.upstream.pipeline_lingbot_video_i2v",
        "yogurt_lingbot_video.upstream.pipeline_lingbot_video",
    ):
        sys.modules.pop(name, None)
    base = importlib.import_module("yogurt_lingbot_video.upstream.pipeline_lingbot_video")
    i2v = importlib.import_module("yogurt_lingbot_video.upstream.pipeline_lingbot_video_i2v")
    return base.LingBotVideoPipeline, i2v.LingBotVideoImageToVideoPipeline


def _pipeline_with_minimal_denoising(pipeline_class, monkeypatch):
    pipeline = object.__new__(pipeline_class)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = _Transformer()
    pipeline.scheduler = _Scheduler()
    pipeline.vae = None
    pipeline._execution_device = torch.device("cpu")
    pipeline.progress_bar = lambda values: values
    pipeline.check_inputs = lambda *args: None
    pipeline.encode_prompt = lambda *args, **kwargs: (torch.zeros(1, 1, 1), torch.ones(1, 1))
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 1, 1, 1)
    pipeline.maybe_free_model_hooks = lambda: None
    monkeypatch.setattr(
        "yogurt_lingbot_video.upstream.pipeline_lingbot_video.compute_refiner_sigmas",
        lambda **kwargs: None,
    )
    return pipeline


def test_generate_frames_wires_comfy_progress_with_requested_step_total(fresh_import, monkeypatch):
    updates = []

    class ProgressBar:
        def __init__(self, total):
            updates.append(("created", total))

        def update_absolute(self, current, total):
            updates.append((current, total))

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = ProgressBar
    comfy_model_management = types.ModuleType("comfy.model_management")
    comfy_model_management.throw_exception_if_processing_interrupted = lambda: None
    comfy.utils = comfy_utils
    comfy.model_management = comfy_model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy_model_management)

    runtime = fresh_import("yogurt_lingbot_video.runtime")

    class Pipeline:
        _execution_device = torch.device("cpu")

        def __call__(self, **kwargs):
            callback = kwargs["step_callback"]
            for current in range(1, kwargs["num_inference_steps"] + 1):
                callback(current, kwargs["num_inference_steps"])
            return SimpleNamespace(frames=np.zeros((1, 1, 4, 4, 3), dtype=np.float32))

    handle = SimpleNamespace(mode="t2v", device=torch.device("cpu"), pipeline=Pipeline(), moe_sequential_cpu_offload=False, moe_block_swap=False)
    runtime.generate_frames(handle, "prompt", "negative", 16, 16, 5, 3, 1.0, 3.0, 1)

    assert updates == [("created", 3), (1, 3), (2, 3), (3, 3)]


def test_base_pipeline_calls_step_callback_once_after_each_scheduler_step(monkeypatch):
    LingBotVideoPipeline, _ = _load_pipeline_types(monkeypatch)
    pipeline = _pipeline_with_minimal_denoising(LingBotVideoPipeline, monkeypatch)
    events = []
    original_step = pipeline.scheduler.step

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        events.append("step")
        return result

    pipeline.scheduler.step = step
    result = pipeline(
        "prompt", height=16, width=16, num_frames=5, num_inference_steps=3,
        guidance_scale=1.0, output_type="latent",
        step_callback=lambda current, total: events.append((current, total)),
    ).frames

    assert events == ["step", (1, 3), "step", (2, 3), "step", (3, 3)]
    assert torch.equal(result, torch.full_like(result, 3))


def test_ti2v_pipeline_calls_step_callback_once_after_each_scheduler_step(monkeypatch):
    _, LingBotVideoImageToVideoPipeline = _load_pipeline_types(monkeypatch)
    pipeline = _pipeline_with_minimal_denoising(LingBotVideoImageToVideoPipeline, monkeypatch)
    pipeline.preprocess_image = lambda *args: torch.zeros(1, 3, 16, 16)
    pipeline._vlm_image = lambda pixel: object()
    pipeline.encode_image_latent = lambda *args, **kwargs: torch.zeros(1, 1, 1, 1, 1)
    pipeline._apply_inpainting = lambda latents, cond: latents
    events = []
    original_step = pipeline.scheduler.step

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        events.append("step")
        return result

    pipeline.scheduler.step = step
    result = pipeline(
        "prompt", image=object(), height=16, width=16, num_frames=5,
        num_inference_steps=3, guidance_scale=1.0, output_type="latent",
        step_callback=lambda current, total: events.append((current, total)),
    ).frames

    assert events == ["step", (1, 3), "step", (2, 3), "step", (3, 3)]
    assert torch.equal(result, torch.full_like(result, 3))
