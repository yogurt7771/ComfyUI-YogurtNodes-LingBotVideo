from types import SimpleNamespace

import torch


def test_generate_disables_manual_vae_offload_for_moe_block_swap_handle(
    fresh_import,
    folder_paths_stub,
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
        cpu_offload=True,
        moe_block_swap=True,
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
    assert calls[0]["offload_vae_during_denoise"] is False
