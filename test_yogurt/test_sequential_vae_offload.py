from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("moe_sequential_cpu_offload", "expected_vae_override"),
    [(True, False), (False, None)],
)
def test_generate_frames_only_disables_denoise_vae_offload_for_moe_sequential_handle(
    fresh_import,
    folder_paths_stub,
    moe_sequential_cpu_offload,
    expected_vae_override,
):
    handles = fresh_import("yogurt_lingbot_video.handles")
    runtime = fresh_import("yogurt_lingbot_video.runtime")
    calls = []

    class Pipeline:
        _execution_device = "cpu"

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=torch.rand(1, 32, 48, 3))

    handle = handles.LingBotVideoPipelineHandle(
        pipeline=Pipeline(),
        model_path="local",
        mode="t2v",
        device="cpu",
        transformer_dtype="float32",
        cpu_offload=moe_sequential_cpu_offload,
        moe_sequential_cpu_offload=moe_sequential_cpu_offload,
    )

    runtime.generate_frames(
        handle,
        prompt="A fox walks through a forest.",
        negative_prompt="negative",
        width=48,
        height=32,
        num_frames=5,
        steps=40,
        guidance_scale=3.5,
        shift=5.0,
        seed=1234,
    )

    assert len(calls) == 1
    call = calls[0]
    if expected_vae_override is None:
        assert "offload_vae_during_denoise" not in call
    else:
        assert call["offload_vae_during_denoise"] is expected_vae_override
    assert call["height"] == 32
    assert call["width"] == 48
    assert call["num_frames"] == 5
    assert call["num_inference_steps"] == 40
    assert call["guidance_scale"] == 3.5
    assert call["shift"] == 5.0
    assert call["generator"].initial_seed() == 1234
