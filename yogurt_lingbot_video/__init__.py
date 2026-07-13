from .model_paths import (
    get_model_root,
    list_local_models,
    list_refiner_models,
    list_rewriter_adapters,
    list_rewriter_base_models,
    list_video_models,
    resolve_model_directory,
)
from .nodes import (
    LingBotVideoGenerate,
    LingBotVideoModelLoader,
    LingBotVideoPromptRewrite,
    LingBotVideoRewriterLoader,
)


NODE_CLASS_MAPPINGS = {
    "YogurtLingBotVideoModelLoader": LingBotVideoModelLoader,
    "YogurtLingBotVideoGenerate": LingBotVideoGenerate,
    "YogurtLingBotVideoRewriterLoader": LingBotVideoRewriterLoader,
    "YogurtLingBotVideoPromptRewrite": LingBotVideoPromptRewrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    node_name: f"{node_class._NODE_NAME} (Yogurt LingBot Video)"
    for node_name, node_class in NODE_CLASS_MAPPINGS.items()
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "get_model_root",
    "list_local_models",
    "list_refiner_models",
    "list_rewriter_adapters",
    "list_rewriter_base_models",
    "list_video_models",
    "resolve_model_directory",
]
