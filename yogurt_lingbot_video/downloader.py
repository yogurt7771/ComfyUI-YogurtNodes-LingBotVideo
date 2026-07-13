import os

try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None

from . import model_paths


DOWNLOADS = {
    "video": {
        model_paths.OFFICIAL_DENSE_MODEL: (
            "robbyant/lingbot-video-dense-1.3b",
            "f9789a7d9b4772a47aba62d4eb5282ddefd1da21",
        ),
        model_paths.OFFICIAL_MOE_MODEL: (
            "robbyant/lingbot-video-moe-30b-a3b",
            "f2e538f64afe00cc4ae674db2aeb52e2945edfd5",
        ),
    },
    "rewriter_base": {
        model_paths.OFFICIAL_REWRITER_BASE: (
            "Qwen/Qwen3.6-27B",
            "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        ),
    },
    "rewriter_adapter": {
        model_paths.OFFICIAL_REWRITER_ADAPTER: (
            "robbyant/lingbot-video-rewriter-lora",
            "dcf6cf2c1fc14ce850b9e5969ab2d8f80c010643",
        ),
    },
}


def _download_destination(model_name: str) -> str:
    parts = model_paths._validate_model_name(model_name)
    root = os.path.realpath(model_paths.get_model_root())
    destination = os.path.realpath(os.path.join(root, *parts))
    if not model_paths._is_within(root, destination):
        raise ValueError(f"Invalid model name: {model_name!r}")
    return destination


def ensure_model_directory(model_name: str, model_kind: str, download_model: bool) -> str:
    try:
        return model_paths.resolve_model_directory(model_name, model_kind)
    except FileNotFoundError:
        pass

    try:
        repository, revision = DOWNLOADS[model_kind][model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported LingBot {model_kind} model: {model_name!r}") from error

    destination = _download_destination(model_name)
    if not download_model:
        raise FileNotFoundError(
            f"LingBot {model_kind} model is missing under {model_paths.get_model_root()}. "
            f"Expected {destination}; copy the model there or enable download_model explicitly."
        )
    if huggingface_hub is None:
        raise ImportError("Explicit LingBot model download requires huggingface_hub.")

    os.makedirs(destination, exist_ok=True)
    huggingface_hub.snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=destination,
    )
    return model_paths.resolve_model_directory(model_name, model_kind)


def ensure_model_files(model_name: str, download_model: bool, model_kind: str = "video") -> str:
    return ensure_model_directory(model_name, model_kind, download_model)


def ensure_video_model(model_name: str, download_model: bool) -> str:
    return ensure_model_directory(model_name, "video", download_model)


def ensure_rewriter_base(model_name: str, download_model: bool) -> str:
    return ensure_model_directory(model_name, "rewriter_base", download_model)


def ensure_rewriter_adapter(model_name: str, download_model: bool) -> str:
    return ensure_model_directory(model_name, "rewriter_adapter", download_model)
