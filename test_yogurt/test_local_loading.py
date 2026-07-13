import importlib
import json
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


DENSE_MODEL = "Robbyant--lingbot-video-dense-1.3b"


def _write_component(directory, checkpoint="diffusion_pytorch_model.safetensors"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / checkpoint).write_bytes(b"test checkpoint placeholder")


def _write_video_model(model_root, name=DENSE_MODEL):
    model = model_root / name
    model.mkdir(parents=True, exist_ok=True)
    (model / "model_index.json").write_text(
        '{"_class_name":"LingBotVideoPipeline"}', encoding="utf-8"
    )
    _write_component(model / "transformer")
    _write_component(model / "vae")
    _write_component(model / "text_encoder", checkpoint="model.safetensors")
    (model / "processor").mkdir(exist_ok=True)
    (model / "processor" / "processor_config.json").write_text("{}", encoding="utf-8")
    (model / "scheduler").mkdir(exist_ok=True)
    (model / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    return model


def _module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
    return module


class _Recorder:
    def __init__(self):
        self.loads = []
        self.events = []
        self.instances = {}

    def component(self, role):
        recorder = self

        class MovableState:
            def __init__(self):
                self.device = torch.device("cpu")

            @property
            def data(self):
                return self

            @data.setter
            def data(self, value):
                self.device = value.device

            def to(self, device):
                moved = MovableState()
                moved.device = torch.device(device)
                return moved

        class Component:
            def __init__(self, *args, **kwargs):
                self.role = role
                self._parameters = {}
                self._buffers = {}
                recorder.instances.setdefault(role, []).append(self)
                recorder.events.append((role, "init", args, kwargs))

            @classmethod
            def from_pretrained(cls, source, *args, **kwargs):
                recorder.loads.append((role, Path(source), kwargs))
                instance = cls()
                if role == "transformer":
                    config_path = Path(source) / kwargs.get("subfolder", "") / "config.json"
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    config.setdefault("num_experts", 0)
                    instance.config = types.SimpleNamespace(**config)
                    instance.blocks = [recorder.component(f"transformer_block_{index}")() for index in range(4)]
                    instance.non_block_modules = [
                        recorder.component("transformer_non_block")()
                    ]
                    instance._parameters = {"direct": MovableState()}
                    instance._buffers = {"direct": MovableState()}
                return instance

            @classmethod
            def from_config(cls, source, *args, **kwargs):
                recorder.loads.append((role, Path(source), kwargs))
                return cls()

            def eval(self):
                return self

            def to(self, device=None, *args, **kwargs):
                recorder.events.append((role, "to", device, kwargs))
                return self

            def requires_grad_(self, value):
                return self

            def named_children(self):
                if self.role != "transformer":
                    return iter(())
                return iter(
                    (
                        ("blocks", self.blocks),
                        ("non_block", self.non_block_modules[0]),
                    )
                )

            def parameters(self, recurse=True):
                return iter(self._parameters.values())

            def buffers(self, recurse=True):
                return iter(self._buffers.values())

        Component.__name__ = f"Fake{role.title().replace('_', '')}"
        return Component

    def pipeline(self, role):
        recorder = self
        component = self.component(role)

        class Pipeline(component):
            @classmethod
            def from_pretrained(cls, source, *args, **kwargs):
                recorder.loads.append((role, Path(source), kwargs))
                return cls()

            def enable_model_cpu_offload(self, *args, **kwargs):
                recorder.events.append((role, "cpu_offload", args, kwargs))
                recorder.events.append((role, "model_cpu_offload", args, kwargs))
                return self

            def enable_sequential_cpu_offload(self, *args, **kwargs):
                recorder.events.append((role, "cpu_offload", args, kwargs))
                recorder.events.append((role, "sequential_cpu_offload", args, kwargs))
                return self

        Pipeline.__name__ = f"Fake{role.title().replace('_', '')}"
        return Pipeline


def _install_fake_dependencies(monkeypatch, recorder):
    transformer = recorder.component("transformer")
    vae = recorder.component("vae")
    text_encoder = recorder.component("text_encoder")
    processor = recorder.component("processor")
    scheduler = recorder.component("scheduler")
    regular = recorder.pipeline("regular_pipeline")
    image = recorder.pipeline("image_pipeline")

    def cpu_offload(model, execution_device=None, offload_buffers=False, preload_module_classes=None):
        recorder.events.append(
            (
                getattr(model, "role", type(model).__name__),
                "accelerate_cpu_offload",
                execution_device,
                {
                    "offload_buffers": offload_buffers,
                    "preload_module_classes": preload_module_classes,
                },
            )
        )
        return model

    _module(monkeypatch, "accelerate", cpu_offload=cpu_offload)

    class AlignDevicesHook:
        def __init__(self, execution_device=None, **kwargs):
            self.execution_device = execution_device
            self.kwargs = kwargs

    def add_hook_to_module(model, hook):
        recorder.events.append(
            (
                getattr(model, "role", type(model).__name__),
                "align_devices_hook",
                hook.execution_device,
                hook.kwargs,
            )
        )
        model._hf_hook = hook
        return model

    _module(
        monkeypatch,
        "accelerate.hooks",
        AlignDevicesHook=AlignDevicesHook,
        add_hook_to_module=add_hook_to_module,
    )

    def apply_group_offloading(
        model,
        *,
        onload_device,
        offload_device,
        offload_type,
        num_blocks_per_group,
    ):
        recorder.events.append(
            (
                getattr(model, "role", type(model).__name__),
                "group_offload",
                onload_device,
                {
                    "offload_device": offload_device,
                    "offload_type": offload_type,
                    "num_blocks_per_group": num_blocks_per_group,
                },
            )
        )

    _module(
        monkeypatch,
        "diffusers",
        AutoencoderKL=vae,
        AutoencoderKLWan=vae,
        FlowMatchEulerDiscreteScheduler=scheduler,
        UniPCMultistepScheduler=scheduler,
    )
    _module(monkeypatch, "diffusers.hooks", apply_group_offloading=apply_group_offloading)
    _module(
        monkeypatch,
        "diffusers.models",
        AutoencoderKL=vae,
        AutoencoderKLWan=vae,
    )
    _module(
        monkeypatch,
        "diffusers.schedulers",
        FlowMatchEulerDiscreteScheduler=scheduler,
        UniPCMultistepScheduler=scheduler,
    )
    _module(
        monkeypatch,
        "transformers",
        AutoModel=text_encoder,
        AutoModelForCausalLM=text_encoder,
        AutoModelForImageTextToText=text_encoder,
        Qwen3VLForConditionalGeneration=text_encoder,
        AutoProcessor=processor,
    )
    _module(
        monkeypatch,
        "lingbot_video",
        LingBotVideoPipeline=regular,
        LingBotVideoImageToVideoPipeline=image,
        LingBotVideoTransformer3DModel=transformer,
        FlowUniPCMultistepScheduler=scheduler,
    )
    _module(
        monkeypatch,
        "lingbot_video.pipeline_lingbot_video",
        LingBotVideoPipeline=regular,
    )
    _module(
        monkeypatch,
        "lingbot_video.pipeline_lingbot_video_i2v",
        LingBotVideoImageToVideoPipeline=image,
        LingBotVideoI2VPipeline=image,
    )
    _module(
        monkeypatch,
        "lingbot_video.transformer_lingbot_video",
        LingBotVideoTransformer3DModel=transformer,
    )
    _module(
        monkeypatch,
        "lingbot_video.scheduling_flow_unipc",
        FlowUniPCMultistepScheduler=scheduler,
    )
    _module(monkeypatch, "yogurt_lingbot_video.upstream")
    _module(
        monkeypatch,
        "yogurt_lingbot_video.upstream.pipeline_lingbot_video",
        LingBotVideoPipeline=regular,
    )
    _module(
        monkeypatch,
        "yogurt_lingbot_video.upstream.pipeline_lingbot_video_i2v",
        LingBotVideoImageToVideoPipeline=image,
        LingBotVideoI2VPipeline=image,
    )
    _module(
        monkeypatch,
        "yogurt_lingbot_video.upstream.transformer_lingbot_video",
        LingBotVideoTransformer3DModel=transformer,
    )
    _module(
        monkeypatch,
        "yogurt_lingbot_video.upstream.scheduling_flow_unipc",
        FlowUniPCMultistepScheduler=scheduler,
    )

    model_management = _module(monkeypatch, "comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cuda")
    model_management.unet_offload_device = lambda: torch.device("cpu")
    model_management.text_encoder_device = lambda: torch.device("cuda")
    model_management.text_encoder_offload_device = lambda: torch.device("cpu")
    model_management.vae_device = lambda: torch.device("cuda")
    model_management.vae_offload_device = lambda: torch.device("cpu")
    model_management.throw_exception_if_processing_interrupted = lambda: None
    model_management.soft_empty_cache = lambda: None
    comfy = _module(monkeypatch, "comfy")
    comfy.model_management = model_management


def _load_plugin():
    for name in list(sys.modules):
        if name == "yogurt_lingbot_video" or name.startswith("yogurt_lingbot_video."):
            sys.modules.pop(name)
    return importlib.import_module("yogurt_lingbot_video")


def _loader(plugin):
    node_class = next(
        node
        for node in plugin.NODE_CLASS_MAPPINGS.values()
        if node.RETURN_TYPES
        and "PIPELINE" in node.RETURN_TYPES[0]
        and "model_name" in node.INPUT_TYPES().get("required", {})
    )
    return node_class()


def _loader_arguments(node, mode, cpu_offload):
    inputs = {
        **node.INPUT_TYPES().get("required", {}),
        **node.INPUT_TYPES().get("optional", {}),
    }
    arguments = {}
    for name, spec in inputs.items():
        if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            arguments[name] = spec[1]["default"]
    mode_key = next(
        name
        for name, spec in inputs.items()
        if isinstance(spec[0], (tuple, list)) and set(spec[0]) == {"t2i", "t2v", "ti2v"}
    )
    offload_key = next(name for name in inputs if "offload" in name.lower())
    arguments["model_name"] = DENSE_MODEL
    arguments[mode_key] = mode
    arguments[offload_key] = cpu_offload
    arguments["download_model"] = False
    for name, spec in inputs.items():
        if "dtype" in name.lower() and isinstance(spec[0], (tuple, list)) and "float32" in spec[0]:
            arguments[name] = "float32"
    if "device" in inputs:
        choices = inputs["device"][0]
        arguments["device"] = "cuda" if "cuda" in choices else choices[0]
    return arguments


def _execute(node, arguments):
    return getattr(node, node.FUNCTION)(**arguments)[0]


def _retained_path(handle):
    for name in ("model_path", "model_directory", "model_dir", "path"):
        value = getattr(handle, name, None)
        if value is not None:
            return Path(value)
    raise AssertionError("loaded handle must retain its selected local model path")


@pytest.mark.parametrize("mode", ["t2i", "t2v"])
def test_regular_modes_load_every_component_from_selected_local_model_only(
    folder_paths_stub, monkeypatch, mode
):
    recorder = _Recorder()
    plugin = _load_plugin()
    _install_fake_dependencies(monkeypatch, recorder)
    paths = importlib.import_module("yogurt_lingbot_video.model_paths")
    selected = _write_video_model(Path(paths.get_model_root()))
    node = _loader(plugin)

    handle = _execute(node, _loader_arguments(node, mode, cpu_offload=False))

    loaded_roles = {role for role, _source, _kwargs in recorder.loads}
    constructed_roles = {role for role, event, *_ in recorder.events if event == "init"}
    assert {"transformer", "vae", "text_encoder", "processor", "scheduler"} <= loaded_roles
    assert "regular_pipeline" in constructed_roles
    assert "image_pipeline" not in constructed_roles
    expected_subfolders = {
        "transformer": "transformer",
        "vae": "vae",
        "text_encoder": "text_encoder",
        "processor": "processor",
        "scheduler": "scheduler",
    }
    for role, folder in expected_subfolders.items():
        sources = [
            (source / kwargs.get("subfolder", "")).resolve()
            for loaded_role, source, kwargs in recorder.loads
            if loaded_role == role
        ]
        assert sources
        assert all(source == (selected / folder).resolve() for source in sources)
    assert all(source.resolve().is_relative_to(selected.resolve()) for _role, source, _kwargs in recorder.loads)
    assert all(kwargs.get("local_files_only", True) is True for _role, _source, kwargs in recorder.loads)
    assert any(role == "regular_pipeline" and event == "to" for role, event, *_ in recorder.events)
    assert not any(role == "regular_pipeline" and event == "cpu_offload" for role, event, *_ in recorder.events)
    assert _retained_path(handle).resolve() == selected.resolve()
    assert handle.mode == mode
    assert str(handle.device) == "cuda"
    assert str(handle.dtype).removeprefix("torch.") == "float32"


def test_ti2v_selects_image_pipeline_and_honors_cpu_offload(folder_paths_stub, monkeypatch):
    recorder = _Recorder()
    plugin = _load_plugin()
    _install_fake_dependencies(monkeypatch, recorder)
    paths = importlib.import_module("yogurt_lingbot_video.model_paths")
    selected = _write_video_model(Path(paths.get_model_root()))
    node = _loader(plugin)

    handle = _execute(node, _loader_arguments(node, "ti2v", cpu_offload=True))

    constructed_roles = {role for role, event, *_ in recorder.events if event == "init"}
    assert "image_pipeline" in constructed_roles
    assert "regular_pipeline" not in constructed_roles
    assert any(role == "image_pipeline" and event == "cpu_offload" for role, event, *_ in recorder.events)
    assert not any(role == "image_pipeline" and event == "to" for role, event, *_ in recorder.events)
    assert all(source.resolve().is_relative_to(selected.resolve()) for _role, source, _kwargs in recorder.loads)
    assert _retained_path(handle).resolve() == selected.resolve()
    assert handle.mode == "ti2v"
    assert str(handle.device) == "cuda"
    assert str(handle.dtype).removeprefix("torch.") == "float32"


@pytest.mark.parametrize("num_experts", [0, 8])
def test_cpu_offload_api_matches_transformer_architecture(
    folder_paths_stub, monkeypatch, num_experts
):
    recorder = _Recorder()
    plugin = _load_plugin()
    _install_fake_dependencies(monkeypatch, recorder)
    paths = importlib.import_module("yogurt_lingbot_video.model_paths")
    selected = _write_video_model(Path(paths.get_model_root()))
    (selected / "transformer" / "config.json").write_text(
        f'{{"num_experts":{num_experts}}}', encoding="utf-8"
    )
    node = _loader(plugin)

    arguments = _loader_arguments(node, "t2v", cpu_offload=True)
    arguments["moe_gpu_blocks"] = 2
    handle = _execute(node, arguments)

    calls = [event for role, event, *_ in recorder.events if role == "regular_pipeline"]
    if num_experts == 0:
        assert "model_cpu_offload" in calls
        assert "sequential_cpu_offload" not in calls
        assert not handle.moe_block_swap
        assert not any(event == "group_offload" for _role, event, *_ in recorder.events)
    else:
        assert "model_cpu_offload" not in calls
        assert "sequential_cpu_offload" not in calls
        assert handle.moe_block_swap
        assert not any(
            role == "transformer" and event == "to"
            for role, event, *_ in recorder.events
        )
        assert all(
            any(
                role == f"transformer_block_{index}" and event == "to" and str(device) == "cuda"
                for role, event, device, _kwargs in recorder.events
            )
            for index in range(2)
        )
        assert any(
            role == "transformer_non_block" and event == "to" and str(device) == "cuda"
            for role, event, device, _kwargs in recorder.events
        )
        transformer = recorder.instances["transformer"][0]
        assert all(str(state.device) == "cuda" for state in transformer._parameters.values())
        assert all(str(state.device) == "cuda" for state in transformer._buffers.values())
        assert not any(event == "group_offload" for _role, event, *_ in recorder.events)
        assert not any(
            role == "transformer" and event == "accelerate_cpu_offload"
            for role, event, *_ in recorder.events
        )
        block_offloads = [
            (role, device, kwargs)
            for role, event, device, kwargs in recorder.events
            if role.startswith("transformer_block_") and event == "accelerate_cpu_offload"
        ]
        assert [role for role, _device, _kwargs in block_offloads] == [
            "transformer_block_2",
            "transformer_block_3",
        ]
        assert all(str(device) == "cuda" for _role, device, _kwargs in block_offloads)
        assert all(
            kwargs["preload_module_classes"] == ["LingBotVideoSparseMoeBlock"]
            for _role, _device, kwargs in block_offloads
        )
        assert all(
            details["preload_module_classes"] in (None, [])
            for role, event, _device, details in recorder.events
            if event == "accelerate_cpu_offload" and not role.startswith("transformer_block_")
        )


def test_loader_exposes_moe_gpu_blocks_with_safe_default_and_range():
    plugin = _load_plugin()
    node = _loader(plugin)

    spec = node.INPUT_TYPES()["required"]["moe_gpu_blocks"]

    assert spec[0] == "INT"
    assert spec[1]["default"] == 12
    assert spec[1]["min"] == 0
    assert spec[1]["max"] == 48


@pytest.mark.parametrize("backend", ["sglang_triton", "sglang_triton_fp8"])
def test_moe_block_swap_rejects_persistent_sglang_expert_cache(
    folder_paths_stub, monkeypatch, backend
):
    recorder = _Recorder()
    plugin = _load_plugin()
    _install_fake_dependencies(monkeypatch, recorder)
    paths = importlib.import_module("yogurt_lingbot_video.model_paths")
    selected = _write_video_model(Path(paths.get_model_root()))
    (selected / "transformer" / "config.json").write_text('{"num_experts":8}', encoding="utf-8")
    monkeypatch.setenv("LINGBOT_MOE_EXPERT_BACKEND", backend)
    node = _loader(plugin)

    with pytest.raises((ValueError, RuntimeError), match="(?i)sglang.*(block swap|offload|cache)|(block swap|offload|cache).*sglang"):
        _execute(node, _loader_arguments(node, "t2v", cpu_offload=True))

    assert not any(
        role == "regular_pipeline" and event in {"model_cpu_offload", "sequential_cpu_offload"}
        for role, event, *_ in recorder.events
    )


def test_video_text_encoder_loader_uses_transformers4_torch_dtype_and_stays_local(
    folder_paths_stub, monkeypatch
):
    recorder = _Recorder()
    plugin = _load_plugin()
    _install_fake_dependencies(monkeypatch, recorder)
    paths = importlib.import_module("yogurt_lingbot_video.model_paths")
    _write_video_model(Path(paths.get_model_root()))
    node = _loader(plugin)

    _execute(node, _loader_arguments(node, "t2i", cpu_offload=False))

    text_encoder_loads = [
        kwargs for role, _source, kwargs in recorder.loads if role == "text_encoder"
    ]
    assert len(text_encoder_loads) == 1
    assert text_encoder_loads[0]["torch_dtype"] is torch.float32
    assert "dtype" not in text_encoder_loads[0]
    assert text_encoder_loads[0]["local_files_only"] is True


def test_rewriter_loader_uses_transformers4_torch_dtype_and_stays_local(
    folder_paths_stub, monkeypatch, tmp_path
):
    recorder = _Recorder()
    _install_fake_dependencies(monkeypatch, recorder)

    class PeftModel:
        @classmethod
        def from_pretrained(cls, model, source, *args, **kwargs):
            recorder.loads.append(("adapter", Path(source), kwargs))
            return model

    _module(monkeypatch, "peft", PeftModel=PeftModel)
    runtime = importlib.import_module("yogurt_lingbot_video.runtime")
    base_model = tmp_path / "Qwen--Qwen3.6-27B"
    adapter_model = tmp_path / "Robbyant--lingbot-video-rewriter-lora"
    base_model.mkdir()
    adapter_model.mkdir()

    runtime.load_rewriter(str(base_model), str(adapter_model))

    text_encoder_loads = [
        kwargs for role, _source, kwargs in recorder.loads if role == "text_encoder"
    ]
    assert len(text_encoder_loads) == 1
    assert text_encoder_loads[0]["torch_dtype"] is torch.bfloat16
    assert "dtype" not in text_encoder_loads[0]
    assert all(kwargs.get("local_files_only") is True for _role, _source, kwargs in recorder.loads)
