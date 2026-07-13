import json
import ntpath
import os
import posixpath

import folder_paths


MODEL_FOLDER_NAME = "lingbot_video"
MODEL_DIRECTORY_NAME = "LingBotVideo"
OFFICIAL_DENSE_MODEL = "Robbyant--lingbot-video-dense-1.3b"
OFFICIAL_MOE_MODEL = "Robbyant--lingbot-video-moe-30b-a3b"
OFFICIAL_REWRITER_BASE = "Qwen--Qwen3.6-27B"
OFFICIAL_REWRITER_ADAPTER = "Robbyant--lingbot-video-rewriter-lora"

_MODEL_ROOT = os.path.join(folder_paths.models_dir, MODEL_DIRECTORY_NAME)
_CHECKPOINT_EXTENSIONS = {".bin", ".ckpt", ".pt", ".pt2", ".pth", ".safetensors", ".sft"}

folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, _MODEL_ROOT, is_default=True)


def get_model_root() -> str:
    return _MODEL_ROOT


def get_registered_roots() -> list[str]:
    return [os.path.realpath(root) for root in folder_paths.get_folder_paths(MODEL_FOLDER_NAME)]


def _is_within(root: str, path: str) -> bool:
    root = os.path.realpath(root)
    path = os.path.realpath(path)
    try:
        return os.path.commonpath((root, path)) == root
    except ValueError:
        return False


def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as source:
            value = json.load(source)
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _has_checkpoint(directory: str) -> bool:
    try:
        return any(
            entry.is_file() and os.path.splitext(entry.name)[1].lower() in _CHECKPOINT_EXTENSIONS
            for entry in os.scandir(directory)
        )
    except OSError:
        return False


def is_video_model(directory: str) -> bool:
    model_index = _load_json(os.path.join(directory, "model_index.json"))
    if not model_index:
        return False
    required_directories = ("processor", "scheduler", "text_encoder", "transformer", "vae")
    if not all(os.path.isdir(os.path.join(directory, name)) for name in required_directories):
        return False
    return _has_checkpoint(os.path.join(directory, "transformer")) and _has_checkpoint(
        os.path.join(directory, "vae")
    )


def has_refiner(directory: str) -> bool:
    refiner = os.path.join(directory, "refiner")
    return _load_json(os.path.join(refiner, "config.json")) is not None and _has_checkpoint(refiner)


def is_rewriter_base(directory: str) -> bool:
    config = _load_json(os.path.join(directory, "config.json"))
    return config is not None and _has_checkpoint(directory)


def is_rewriter_adapter(directory: str) -> bool:
    config = _load_json(os.path.join(directory, "adapter_config.json"))
    return config is not None and _has_checkpoint(directory)


def _list_directories(predicate, skip_video_components: bool = False) -> list[str]:
    models = set()
    for root in get_registered_roots():
        if not os.path.isdir(root):
            continue
        for directory, subdirectories, _filenames in os.walk(root):
            subdirectories[:] = [
                name for name in subdirectories if _is_within(root, os.path.join(directory, name))
            ]
            if directory == root or not _is_within(root, directory):
                continue
            if skip_video_components and is_video_model(directory):
                subdirectories[:] = []
                continue
            if predicate(directory):
                models.add(os.path.relpath(directory, root).replace(os.sep, "/"))
                subdirectories[:] = []
    return sorted(models)


def list_video_models() -> list[str]:
    return _list_directories(is_video_model)


def list_refiner_models() -> list[str]:
    return _list_directories(lambda directory: is_video_model(directory) and has_refiner(directory))


def list_rewriter_base_models() -> list[str]:
    return _list_directories(is_rewriter_base, skip_video_components=True)


def list_rewriter_adapters() -> list[str]:
    return _list_directories(is_rewriter_adapter, skip_video_components=True)


def list_local_models(model_kind: str = "video") -> list[str]:
    listings = {
        "video": list_video_models,
        "refiner": list_refiner_models,
        "rewriter_base": list_rewriter_base_models,
        "rewriter_adapter": list_rewriter_adapters,
    }
    try:
        return listings[model_kind]()
    except KeyError as error:
        raise ValueError(f"Unsupported LingBot model kind: {model_kind!r}") from error


def _validate_model_name(model_name: str) -> list[str]:
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Model name must be a non-empty relative path")
    normalized_name = model_name.replace("\\", "/")
    parts = normalized_name.split("/")
    if (
        posixpath.isabs(model_name)
        or ntpath.isabs(model_name)
        or ntpath.splitdrive(model_name)[0]
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"Invalid model name: {model_name!r}")
    return parts


def resolve_model_directory(model_name: str, model_kind: str = "video") -> str:
    predicates = {
        "video": is_video_model,
        "refiner": lambda directory: is_video_model(directory) and has_refiner(directory),
        "rewriter_base": is_rewriter_base,
        "rewriter_adapter": is_rewriter_adapter,
    }
    try:
        predicate = predicates[model_kind]
    except KeyError as error:
        raise ValueError(f"Unsupported LingBot model kind: {model_kind!r}") from error

    parts = _validate_model_name(model_name)
    for root in get_registered_roots():
        directory = os.path.realpath(os.path.join(root, *parts))
        if _is_within(root, directory) and predicate(directory):
            return directory
    raise FileNotFoundError(f"LingBot {model_kind} model not found: {model_name!r}")


def resolve_video_model_directory(model_name: str) -> str:
    return resolve_model_directory(model_name, "video")


def resolve_rewriter_base_directory(model_name: str) -> str:
    return resolve_model_directory(model_name, "rewriter_base")


def resolve_rewriter_adapter_directory(model_name: str) -> str:
    return resolve_model_directory(model_name, "rewriter_adapter")
