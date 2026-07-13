import os
import sys
import types
from pathlib import Path

import pytest


DENSE_MODEL = "Robbyant--lingbot-video-dense-1.3b"
MOE_MODEL = "Robbyant--lingbot-video-moe-30b-a3b"
REWRITER_BASE = "Qwen--Qwen3.6-27B"
REWRITER_ADAPTER = "Robbyant--lingbot-video-rewriter-lora"


def _write_component(directory, checkpoint="diffusion_pytorch_model.safetensors"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / checkpoint).write_bytes(b"test checkpoint placeholder")


def _write_video_model(model_root, name, with_refiner=False):
    model = model_root / name
    model.mkdir(parents=True, exist_ok=True)
    (model / "model_index.json").write_text(
        '{"_class_name":"LingBotVideoPipeline"}', encoding="utf-8"
    )
    _write_component(model / "transformer")
    _write_component(model / "vae")
    _write_component(model / "text_encoder", checkpoint="model.safetensors")
    (model / "processor").mkdir()
    (model / "processor" / "processor_config.json").write_text("{}", encoding="utf-8")
    (model / "scheduler").mkdir()
    (model / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    if with_refiner:
        _write_component(model / "refiner")
    return model


def _write_rewriter_base(model_root, name=REWRITER_BASE):
    model = model_root / name
    _write_component(model, checkpoint="model.safetensors")
    (model / "processor_config.json").write_text("{}", encoding="utf-8")
    return model


def _write_rewriter_adapter(model_root, name=REWRITER_ADAPTER):
    model = model_root / name
    model.mkdir(parents=True, exist_ok=True)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model / "adapter_model.safetensors").write_bytes(b"test adapter placeholder")
    return model


def test_model_root_comes_from_comfy_models_dir_and_registers_folder(fresh_import, folder_paths_stub):
    paths = fresh_import("yogurt_lingbot_video.model_paths")

    model_root = Path(paths.get_model_root()).resolve()
    comfy_root = folder_paths_stub.model_dir.resolve()
    assert model_root.is_relative_to(comfy_root)
    assert model_root != comfy_root
    assert any(
        path.resolve() == model_root and is_default
        for _name, path, is_default in folder_paths_stub.registrations
    )


def test_discovers_dense_moe_and_separate_rewriter_layouts(fresh_import, folder_paths_stub):
    paths = fresh_import("yogurt_lingbot_video.model_paths")
    model_root = Path(paths.get_model_root())
    _write_video_model(model_root, DENSE_MODEL)
    _write_video_model(model_root, MOE_MODEL, with_refiner=True)
    _write_rewriter_base(model_root)
    _write_rewriter_adapter(model_root)
    (model_root / "unfinished").mkdir()
    (model_root / "unfinished" / "model_index.json").write_text("{}", encoding="utf-8")
    assert paths.list_local_models("video") == [DENSE_MODEL, MOE_MODEL]
    assert paths.list_local_models("rewriter_base") == [REWRITER_BASE]
    assert paths.list_local_models("rewriter_adapter") == [REWRITER_ADAPTER]


def test_resolves_only_relative_paths_inside_registered_model_roots(fresh_import, folder_paths_stub):
    paths = fresh_import("yogurt_lingbot_video.model_paths")
    model_root = Path(paths.get_model_root())
    dense = _write_video_model(model_root, DENSE_MODEL)
    base = _write_rewriter_base(model_root)
    adapter = _write_rewriter_adapter(model_root)

    assert Path(paths.resolve_video_model_directory(DENSE_MODEL)).resolve() == dense.resolve()
    assert Path(paths.resolve_rewriter_base_directory(REWRITER_BASE)).resolve() == base.resolve()
    assert Path(paths.resolve_rewriter_adapter_directory(REWRITER_ADAPTER)).resolve() == adapter.resolve()

    resolvers = (
        paths.resolve_video_model_directory,
        paths.resolve_rewriter_base_directory,
        paths.resolve_rewriter_adapter_directory,
    )
    invalid_names = ("../outside", "..\\outside", "/tmp/outside", "C:\\outside")
    for resolver in resolvers:
        for invalid_name in invalid_names:
            with pytest.raises((ValueError, FileNotFoundError)):
                resolver(invalid_name)


def test_missing_assets_with_download_disabled_are_actionable_and_offline(
    fresh_import, folder_paths_stub, monkeypatch
):
    hub = types.ModuleType("huggingface_hub")

    def unexpected_download(*args, **kwargs):
        raise AssertionError("download_model=False must not attempt a download")

    hub.snapshot_download = unexpected_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    downloader = fresh_import("yogurt_lingbot_video.downloader")

    with pytest.raises((FileNotFoundError, RuntimeError)) as video_error:
        downloader.ensure_video_model(DENSE_MODEL, download_model=False)
    with pytest.raises((FileNotFoundError, RuntimeError)) as base_error:
        downloader.ensure_rewriter_base(REWRITER_BASE, download_model=False)
    with pytest.raises((FileNotFoundError, RuntimeError)) as adapter_error:
        downloader.ensure_rewriter_adapter(REWRITER_ADAPTER, download_model=False)

    expected_root = str(folder_paths_stub.registrations[-1][1].resolve()).lower()
    for error in (video_error, base_error, adapter_error):
        message = str(error.value).lower()
        assert expected_root in message
        assert "download_model" in message


def test_explicit_downloads_use_comfy_model_root_as_local_destination(
    fresh_import, folder_paths_stub, monkeypatch
):
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(*args, **kwargs):
        repo_id = kwargs.get("repo_id", args[0] if args else None)
        local_dir = Path(kwargs["local_dir"])
        calls.append((repo_id, local_dir, kwargs.get("cache_dir")))
        normalized_repo = repo_id.lower()
        if normalized_repo == "robbyant/lingbot-video-dense-1.3b":
            _write_video_model(local_dir.parent, local_dir.name)
        elif normalized_repo == "qwen/qwen3.6-27b":
            _write_rewriter_base(local_dir.parent, local_dir.name)
        elif normalized_repo == "robbyant/lingbot-video-rewriter-lora":
            _write_rewriter_adapter(local_dir.parent, local_dir.name)
        else:
            raise AssertionError(f"unexpected repository: {repo_id}")
        return str(local_dir)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    downloader = fresh_import("yogurt_lingbot_video.downloader")

    video = downloader.ensure_video_model(DENSE_MODEL, download_model=True)
    base = downloader.ensure_rewriter_base(REWRITER_BASE, download_model=True)
    adapter = downloader.ensure_rewriter_adapter(REWRITER_ADAPTER, download_model=True)

    model_root = folder_paths_stub.registrations[-1][1].resolve()
    assert Path(video).resolve().is_relative_to(model_root)
    assert Path(base).resolve().is_relative_to(model_root)
    assert Path(adapter).resolve().is_relative_to(model_root)
    assert {repo_id.lower() for repo_id, *_ in calls} == {
        "robbyant/lingbot-video-dense-1.3b",
        "qwen/qwen3.6-27b",
        "robbyant/lingbot-video-rewriter-lora",
    }
    assert all(local_dir.resolve().is_relative_to(model_root) for _, local_dir, _ in calls)
    assert all(cache_dir is None or Path(cache_dir).resolve().is_relative_to(model_root) for *_, cache_dir in calls)
    assert not any(
        token in str(local_dir).lower()
        for _, local_dir, _ in calls
        for token in ("huggingface\\hub", "huggingface/hub", ".cache\\huggingface", ".cache/huggingface")
    )
