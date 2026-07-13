import contextlib
import gc
import json
import os
from typing import Any

from .handles import LingBotVideoPipelineHandle, LingBotVideoRewriterHandle


_VIDEO_MODES = {"t2i", "t2v", "ti2v"}
_PROMPT_RUNTIME_KEYS = {"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"}


def _device_to_string(device: Any) -> str:
    if getattr(device, "index", None) is None:
        return str(device)
    return f"{device.type}:{device.index}"


def _normalize_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("LingBot prompt must not be empty.")
    try:
        sample = json.loads(prompt)
    except json.JSONDecodeError:
        sample = {"comprehensive_description": prompt}

    if isinstance(sample, dict):
        caption = sample.get("caption")
        if caption is None:
            caption = {key: value for key, value in sample.items() if key not in _PROMPT_RUNTIME_KEYS}
    else:
        caption = sample
    if isinstance(caption, (dict, list)):
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    return str(caption)


def _resolve_dtype(dtype_name: str, device: Any):
    import torch

    if dtype_name == "auto":
        import comfy.model_management

        if comfy.model_management.should_use_bf16(device):
            return torch.bfloat16
        if getattr(device, "type", "cpu") != "cpu":
            return torch.float16
        return torch.float32

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtypes[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported transformer dtype: {dtype_name!r}") from error


def _require_runtime_dependencies() -> None:
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "LingBot Video requires the dependencies listed in this plugin's requirements.txt."
        ) from error


@contextlib.contextmanager
def _official_math_settings(torch):
    reduction = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(reduction)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision(matmul_precision)


def _enable_moe_block_swap(pipeline, device, transformer, text_encoder, vae, gpu_blocks: int) -> None:
    from accelerate import cpu_offload
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    backend = os.environ.get("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm").lower().strip()
    if backend in {"sglang_triton", "triton", "sglang", "sglang_triton_fp8", "triton_fp8", "sglang_fp8"}:
        raise ValueError(
            f"LINGBOT_MOE_EXPERT_BACKEND={backend!r} is incompatible with MoE block swap; "
            "use grouped_mm."
        )

    pipeline._offload_device = device
    pipeline._offload_gpu_id = getattr(device, "index", None) or 0
    blocks = transformer.blocks
    gpu_blocks = min(max(int(gpu_blocks), 0), len(blocks))
    for block in blocks[:gpu_blocks]:
        block.to(device)
    for block in blocks[gpu_blocks:]:
        cpu_offload(
            block,
            device,
            offload_buffers=len(block._parameters) > 0,
            preload_module_classes=["LingBotVideoSparseMoeBlock"],
        )
    for name, module in transformer.named_children():
        if name != "blocks":
            module.to(device)
    for parameter in transformer.parameters(recurse=False):
        parameter.data = parameter.data.to(device)
    for buffer in transformer.buffers(recurse=False):
        buffer.data = buffer.data.to(device)
    add_hook_to_module(transformer, AlignDevicesHook(execution_device=device))
    cpu_offload(text_encoder, device, offload_buffers=len(text_encoder._parameters) > 0)
    cpu_offload(vae, device, offload_buffers=len(vae._parameters) > 0)


def load_video_pipeline(
    model_path: str,
    mode: str,
    transformer_dtype: str,
    cpu_offload: bool,
    moe_gpu_blocks: int = 12,
) -> LingBotVideoPipelineHandle:
    if mode not in _VIDEO_MODES:
        raise ValueError(f"Unsupported LingBot generation mode: {mode!r}")

    _require_runtime_dependencies()
    import comfy.model_management
    import torch
    from diffusers import AutoencoderKLWan
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from .upstream.pipeline_lingbot_video import LingBotVideoPipeline
    from .upstream.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline
    from .upstream.scheduling_flow_unipc import FlowUniPCMultistepScheduler
    from .upstream.transformer_lingbot_video import LingBotVideoTransformer3DModel

    comfy.model_management.throw_exception_if_processing_interrupted()
    device = comfy.model_management.get_torch_device()
    transformer_torch_dtype = _resolve_dtype(transformer_dtype, device)
    text_encoder_dtype = torch.bfloat16 if transformer_torch_dtype != torch.float32 else torch.float32

    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=transformer_torch_dtype,
        local_files_only=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=text_encoder_dtype,
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        subfolder="processor",
        local_files_only=True,
        trust_remote_code=True,
    )
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        local_files_only=True,
    )
    pipeline_class = LingBotVideoImageToVideoPipeline if mode == "ti2v" else LingBotVideoPipeline
    pipeline = pipeline_class(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        scheduler=scheduler,
    )
    moe_block_swap = bool(cpu_offload and transformer.config.num_experts > 0)
    if cpu_offload:
        if moe_block_swap:
            _enable_moe_block_swap(
                pipeline, device, transformer, text_encoder, vae, moe_gpu_blocks
            )
        else:
            pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)
    comfy.model_management.soft_empty_cache()
    return LingBotVideoPipelineHandle(
        pipeline=pipeline,
        model_path=model_path,
        mode=mode,
        device=_device_to_string(device),
        transformer_dtype=str(transformer_torch_dtype).replace("torch.", ""),
        cpu_offload=bool(cpu_offload),
        moe_block_swap=moe_block_swap,
    )


def _validate_generation(mode: str, width: int, height: int, num_frames: int, image: Any) -> int:
    if mode not in _VIDEO_MODES:
        raise ValueError(f"Unsupported LingBot generation mode: {mode!r}")
    if width % 16 or height % 16:
        raise ValueError(f"LingBot width and height must be multiples of 16, got {width}x{height}.")
    if mode == "t2i":
        return 1
    if num_frames != 1 and (num_frames - 1) % 4:
        raise ValueError(f"LingBot num_frames must be 1 or 4n+1, got {num_frames}.")
    if mode == "ti2v" and image is None:
        raise ValueError("LingBot TI2V requires an IMAGE input.")
    return num_frames


def _to_pil_image(image: Any):
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a Comfy IMAGE tensor, got shape {tuple(array.shape)}.")
    if array.shape[-1] not in {1, 3, 4} and array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array.astype(np.float32), 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _unwrap_frames(value: Any) -> Any:
    for attribute in ("frames", "images", "videos"):
        if hasattr(value, attribute):
            return getattr(value, attribute)
    return value


def normalize_frames(value: Any):
    import numpy as np
    import torch
    from PIL import Image

    value = _unwrap_frames(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("LingBot returned no frames.")
        if len(value) == 1:
            value = value[0]
        elif all(isinstance(frame, Image.Image) for frame in value):
            value = np.stack([np.asarray(frame.convert("RGB")) for frame in value])
        else:
            frames = [normalize_frames(frame) for frame in value]
            return torch.cat(frames, dim=0)

    if isinstance(value, Image.Image):
        value = np.asarray(value.convert("RGB"))
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Unsupported LingBot frame output: {type(value)!r}")

    output = value.detach().cpu()
    if output.ndim == 5:
        if output.shape[-1] in {1, 3, 4}:
            output = output[0]
        elif output.shape[1] in {1, 3, 4}:
            output = output[0].permute(1, 2, 3, 0)
        elif output.shape[2] in {1, 3, 4}:
            output = output[0].permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Unexpected LingBot 5D frame shape: {tuple(output.shape)}")
    if output.ndim == 3:
        output = output.unsqueeze(0)
    if output.ndim != 4:
        raise ValueError(f"Unexpected LingBot frame shape: {tuple(output.shape)}")
    if output.shape[-1] not in {1, 3, 4} and output.shape[1] in {1, 3, 4}:
        output = output.permute(0, 2, 3, 1)
    if output.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Unsupported LingBot channel layout: {tuple(output.shape)}")
    if output.dtype == torch.uint8:
        output = output.float() / 255.0
    else:
        output = output.float().clamp(0.0, 1.0)
    if output.shape[-1] == 1:
        output = output.repeat(1, 1, 1, 3)
    elif output.shape[-1] == 4:
        output = output[..., :3]
    return output


def generate_frames(
    handle: LingBotVideoPipelineHandle,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    guidance_scale: float,
    shift: float,
    seed: int,
    image: Any = None,
):
    import torch

    from .upstream.defaults import DEFAULT_NEGATIVE_PROMPT, DEFAULT_NEGATIVE_PROMPT_IMAGE

    mode = handle.mode
    prompt = _normalize_prompt(prompt)
    num_frames = _validate_generation(mode, width, height, num_frames, image)
    if not negative_prompt.strip():
        negative_prompt = DEFAULT_NEGATIVE_PROMPT_IMAGE if mode == "t2i" else DEFAULT_NEGATIVE_PROMPT
    step_callback = None
    try:
        import comfy.model_management
        from comfy.utils import ProgressBar

        comfy.model_management.throw_exception_if_processing_interrupted()
        progress_bar = ProgressBar(steps)

        def step_callback(current, total):
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress_bar.update_absolute(current, total)
    except ImportError:
        pass

    device = getattr(handle.pipeline, "_execution_device", handle.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "generator": generator,
        "output_type": "np",
    }
    if mode == "ti2v":
        call_kwargs["image"] = _to_pil_image(image)
    if handle.moe_sequential_cpu_offload or handle.moe_block_swap:
        call_kwargs["offload_vae_during_denoise"] = False
    if step_callback is not None:
        call_kwargs["step_callback"] = step_callback
    with _official_math_settings(torch):
        output = handle.pipeline(**call_kwargs)
    return normalize_frames(output)


class _TransformersRewriterBackend:
    def __init__(self, processor: Any, model: Any, model_loader=None):
        self.processor = processor
        self.model = model
        self.model_loader = model_loader

    def generate(self, text: str, image: Any, use_lora: bool) -> str:
        if self.model is None:
            if self.model_loader is None:
                raise RuntimeError("LingBot rewriter model was released and cannot be reloaded.")
            self.model = self.model_loader()
        content = ([{"type": "image", "image": image}] if image is not None else []) + [
            {"type": "text", "text": text}
        ]
        messages = [{"role": "user", "content": content}]
        chat = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat],
            images=([image] if image is not None else None),
            return_tensors="pt",
        ).to(self.model.device)
        adapter_context = contextlib.nullcontext() if use_lora else self.model.disable_adapter()
        with adapter_context:
            output = self.model.generate(**inputs, max_new_tokens=6144, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def release_vram(self) -> None:
        import comfy.model_management
        from accelerate.hooks import remove_hook_from_submodules

        model = self.model
        if model is None:
            return
        remove_hook_from_submodules(model)
        model.to("cpu")
        self.model = None
        del model
        gc.collect()
        comfy.model_management.soft_empty_cache()


def load_rewriter(base_model_path: str, adapter_model_path: str) -> LingBotVideoRewriterHandle:
    _require_runtime_dependencies()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as error:
        raise ImportError("LingBot prompt rewriting requires transformers and peft.") from error

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    def load_model():
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
        return PeftModel.from_pretrained(model, adapter_model_path, local_files_only=True).eval()

    model = load_model()
    return LingBotVideoRewriterHandle(
        backend=_TransformersRewriterBackend(processor, model, model_loader=load_model),
        base_model_path=base_model_path,
        adapter_model_path=adapter_model_path,
    )


def _step_one_prompt(mode: str, prompt: str, duration: float) -> str:
    if mode == "t2i":
        return f"Expand this image-generation request into a detailed visual caption:\n\n{prompt}"
    return f"Expand this video-generation request into a detailed visual caption for {duration:g} seconds:\n\n{prompt}"


def _step_two_prompt(mode: str, detailed: str, duration: float) -> str:
    duration_line = "" if mode == "t2i" else f"\nVideo duration: {duration:g} seconds."
    return f"Map this detailed LingBot {mode} caption to a JSON object only.{duration_line}\n\n{detailed}"


def _structured_json(raw: Any) -> str:
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            parsed = repair_json(text, return_objects=True)
        except (ImportError, ValueError):
            raise ValueError("LingBot rewriter did not return valid JSON.") from None
    if not isinstance(parsed, dict):
        raise ValueError("LingBot rewriter did not return a JSON object.")
    return json.dumps(parsed, ensure_ascii=False)


def rewrite_prompt(
    handle: LingBotVideoRewriterHandle,
    prompt: str,
    mode: str,
    duration: float,
    image: Any = None,
    release_rewriter_after_rewrite: bool = False,
) -> tuple[str, str]:
    if mode not in _VIDEO_MODES:
        raise ValueError(f"Unsupported LingBot rewrite mode: {mode!r}")
    if mode == "ti2v" and image is None:
        raise ValueError("LingBot TI2V prompt rewriting requires an IMAGE input.")
    first_frame = _to_pil_image(image) if image is not None else None
    try:
        detailed = handle.backend.generate(_step_one_prompt(mode, prompt, duration), first_frame, use_lora=False).strip()
        raw_json = handle.backend.generate(_step_two_prompt(mode, detailed, duration), first_frame, use_lora=True)
        result = (_structured_json(raw_json), detailed)
    except BaseException:
        if release_rewriter_after_rewrite:
            try:
                handle.backend.release_vram()
            except Exception:
                pass
        raise
    if release_rewriter_after_rewrite:
        handle.backend.release_vram()
    return result
