import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

class _LayoutCheckingVae(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0))
        self.config = SimpleNamespace(latents_mean=[0.0] * channels, latents_std=[1.0] * channels)
        self.received = None

    def decode(self, latents):
        self.received = latents.clone()
        assert latents.layout == torch.strided
        assert latents.is_contiguous(memory_format=torch.channels_last_3d)
        return SimpleNamespace(sample=latents)


def _load_pipeline(monkeypatch):
    diffusers = ModuleType("diffusers")
    diffusers.DiffusionPipeline = type("DiffusionPipeline", (), {})
    diffusers_utils = ModuleType("diffusers.utils")
    diffusers_utils.BaseOutput = object
    torch_utils = ModuleType("diffusers.utils.torch_utils")
    torch_utils.randn_tensor = torch.randn
    scheduler = ModuleType("yogurt_lingbot_video.upstream.scheduling_flow_unipc")
    scheduler.FlowUniPCMultistepScheduler = type("FlowUniPCMultistepScheduler", (), {})
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)
    monkeypatch.setitem(sys.modules, "diffusers.utils.torch_utils", torch_utils)
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    sys.modules.pop("yogurt_lingbot_video.upstream.pipeline_lingbot_video", None)
    return importlib.import_module("yogurt_lingbot_video.upstream.pipeline_lingbot_video").LingBotVideoPipeline


def test_decode_passes_5d_latents_to_vae_in_official_channels_last_3d_layout(monkeypatch):
    pipeline_type = _load_pipeline(monkeypatch)
    latents = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
    vae = _LayoutCheckingVae(channels=latents.shape[1])
    pipeline = pipeline_type.__new__(pipeline_type)
    pipeline.vae = vae

    pipeline._decode_latents(latents)

    assert vae.received.shape == latents.shape
    assert vae.received.dtype == latents.dtype
    assert vae.received.device == latents.device
    assert vae.received.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(vae.received, latents)


@pytest.mark.parametrize(
    ("execution_device", "expected_device"),
    [(torch.device("cuda:0"), torch.device("cuda:0")), (None, torch.device("cpu"))],
)
def test_module_device_prefers_root_cpu_offload_hook_then_falls_back_to_parameter(
    monkeypatch, execution_device, expected_device
):
    _load_pipeline(monkeypatch)
    pipeline_module = sys.modules["yogurt_lingbot_video.upstream.pipeline_lingbot_video"]
    module = torch.nn.Linear(1, 1, device="cpu")
    if execution_device is not None:
        module._hf_hook = SimpleNamespace(execution_device=execution_device)

    assert pipeline_module._module_device(module) == expected_device


def test_module_device_finds_first_nested_cpu_offload_execution_device(monkeypatch):
    _load_pipeline(monkeypatch)
    pipeline_module = sys.modules["yogurt_lingbot_video.upstream.pipeline_lingbot_video"]
    first = torch.nn.Linear(1, 1, device="meta")
    second = torch.nn.Linear(1, 1, device="meta")
    first._hf_hook = SimpleNamespace(execution_device=torch.device("cuda:1"))
    second._hf_hook = SimpleNamespace(execution_device=torch.device("cuda:2"))
    module = torch.nn.Sequential(first, second)

    assert pipeline_module._module_device(module) == torch.device("cuda:1")


def test_decode_detaches_frames_only_when_converting_to_numpy(monkeypatch):
    pipeline_type = _load_pipeline(monkeypatch)
    latents = torch.linspace(-1.0, 1.0, 2 * 3 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2, 2)
    latents.requires_grad_(True)
    pipeline = pipeline_type.__new__(pipeline_type)
    pipeline.vae = _LayoutCheckingVae(channels=latents.shape[1])

    videos = pipeline._decode_latents(latents)

    expected = ((latents.detach() + 1.0) / 2.0).permute(0, 2, 3, 4, 1).numpy()
    assert len(videos) == expected.shape[0]
    assert all(video.shape == expected[index].shape for index, video in enumerate(videos))
    assert all((video == expected[index]).all() for index, video in enumerate(videos))
