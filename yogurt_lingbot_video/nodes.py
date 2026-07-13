from . import downloader, model_paths
from .handles import (
    LINGBOT_VIDEO_PIPELINE_TYPE,
    LINGBOT_VIDEO_REWRITER_TYPE,
    LingBotVideoPipelineHandle,
    LingBotVideoRewriterHandle,
)


def _choices(local: list[str], official: list[str]) -> list[str]:
    return list(dict.fromkeys(local + official))


class LingBotVideoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    _choices(
                        model_paths.list_video_models(),
                        [model_paths.OFFICIAL_DENSE_MODEL, model_paths.OFFICIAL_MOE_MODEL],
                    ),
                    {"tooltip": "Select a LingBot diffusers model folder from models/LingBotVideo."},
                ),
                "mode": (["t2v", "ti2v", "t2i"], {"default": "t2v"}),
                "transformer_dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto"},
                ),
                "cpu_offload": ("BOOLEAN", {"default": True}),
                "moe_gpu_blocks": ("INT", {"default": 12, "min": 0, "max": 48, "step": 1}),
                "download_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Explicitly download the selected official model into models/LingBotVideo.",
                    },
                ),
            }
        }

    RETURN_TYPES = (LINGBOT_VIDEO_PIPELINE_TYPE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    OUTPUT_NODE = False
    _NODE_NAME = "Load LingBot Video Model"
    CATEGORY = "YogurtLingBotVideo/Video"
    DESCRIPTION = "Load a local LingBot Video Dense or MoE pipeline. Models are never downloaded unless requested."

    def load(
        self,
        model_name: str,
        mode: str,
        transformer_dtype: str,
        cpu_offload: bool,
        download_model: bool,
        moe_gpu_blocks: int = 12,
    ):
        model_path = downloader.ensure_video_model(model_name, download_model)
        from .runtime import load_video_pipeline

        return (
            load_video_pipeline(
                model_path=model_path,
                mode=mode,
                transformer_dtype=transformer_dtype,
                cpu_offload=cpu_offload,
                moe_gpu_blocks=moe_gpu_blocks,
            ),
        )


class LingBotVideoGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": (LINGBOT_VIDEO_PIPELINE_TYPE,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1001, "step": 4}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "randomize"},
                ),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    OUTPUT_NODE = False
    _NODE_NAME = "LingBot Video Generate"
    CATEGORY = "YogurtLingBotVideo/Video"
    DESCRIPTION = "Generate an image or video with a loaded LingBot Video pipeline."

    def generate(
        self,
        pipeline: LingBotVideoPipelineHandle,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        guidance_scale: float,
        shift: float,
        seed: int,
        image=None,
    ):
        from .runtime import generate_frames

        return (
            generate_frames(
                handle=pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                shift=shift,
                seed=seed,
                image=image,
            ),
        )


class LingBotVideoRewriterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (
                    _choices(
                        model_paths.list_rewriter_base_models(),
                        [model_paths.OFFICIAL_REWRITER_BASE],
                    ),
                ),
                "adapter_model": (
                    _choices(
                        model_paths.list_rewriter_adapters(),
                        [model_paths.OFFICIAL_REWRITER_ADAPTER],
                    ),
                ),
                "download_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Explicitly download the Qwen base and LingBot LoRA into models/LingBotVideo.",
                    },
                ),
            }
        }

    RETURN_TYPES = (LINGBOT_VIDEO_REWRITER_TYPE,)
    RETURN_NAMES = ("rewriter",)
    FUNCTION = "load"
    OUTPUT_NODE = False
    _NODE_NAME = "Load LingBot Prompt Rewriter"
    CATEGORY = "YogurtLingBotVideo/Prompt"
    DESCRIPTION = "Load the Qwen3.6 base model and LingBot prompt-rewriter LoRA."

    def load(self, base_model: str, adapter_model: str, download_model: bool):
        base_path = downloader.ensure_rewriter_base(base_model, download_model)
        adapter_path = downloader.ensure_rewriter_adapter(adapter_model, download_model)
        from .runtime import load_rewriter

        return (load_rewriter(base_path, adapter_path),)


class LingBotVideoPromptRewrite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rewriter": (LINGBOT_VIDEO_REWRITER_TYPE,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["t2v", "ti2v", "t2i"], {"default": "t2v"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "release_rewriter_after_rewrite": ("BOOLEAN", {"default": True}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_json", "detailed_prompt")
    FUNCTION = "rewrite"
    OUTPUT_NODE = False
    _NODE_NAME = "LingBot Prompt Rewrite"
    CATEGORY = "YogurtLingBotVideo/Prompt"
    DESCRIPTION = "Expand a prompt and map it to LingBot's structured JSON caption."

    def rewrite(
        self,
        rewriter: LingBotVideoRewriterHandle,
        prompt: str,
        mode: str,
        duration: float,
        release_rewriter_after_rewrite: bool = True,
        image=None,
    ):
        from .runtime import rewrite_prompt

        return rewrite_prompt(
            rewriter,
            prompt,
            mode,
            duration,
            image,
            release_rewriter_after_rewrite=release_rewriter_after_rewrite,
        )
