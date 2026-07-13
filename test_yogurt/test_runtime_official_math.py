import sys
from types import SimpleNamespace

import pytest
import torch


def _math_state():
    return (
        torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    )


def _set_math_state(reduction_math_sdp, allow_tf32, matmul_precision):
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduction_math_sdp)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision(matmul_precision)


def _generate(generate_frames, handle):
    return generate_frames(
        handle,
        prompt="official math contract",
        negative_prompt="negative",
        width=832,
        height=480,
        num_frames=1,
        steps=2,
        guidance_scale=3.0,
        shift=3.0,
        seed=42,
    )


@pytest.mark.parametrize("pipeline_raises", [False, True])
def test_generate_frames_scopes_official_math_settings(
    repo_root, monkeypatch, pipeline_raises
):
    folder_paths = SimpleNamespace(
        models_dir=str(repo_root / "models"),
        folder_names_and_paths={},
        add_model_folder_path=lambda *args, **kwargs: None,
        get_folder_paths=lambda name: [],
    )
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    sys.path.insert(0, str(repo_root))
    original = _math_state()
    prior = (True, False, "highest")
    observed = []
    try:
        _set_math_state(*prior)
        from yogurt_lingbot_video.handles import LingBotVideoPipelineHandle
        from yogurt_lingbot_video.runtime import generate_frames

        class Pipeline:
            _execution_device = "cpu"

            def __call__(self, **kwargs):
                observed.append(_math_state())
                if pipeline_raises:
                    raise RuntimeError("pipeline failed")
                return SimpleNamespace(images=torch.rand(1, 480, 832, 3))

        handle = LingBotVideoPipelineHandle(
            Pipeline(), "local", "t2i", "cpu", "float32", False
        )
        if pipeline_raises:
            with pytest.raises(RuntimeError, match="pipeline failed"):
                _generate(generate_frames, handle)
        else:
            _generate(generate_frames, handle)

        assert observed == [(False, True, "high")]
        assert _math_state() == prior
    finally:
        _set_math_state(*original)
        sys.path.remove(str(repo_root))
