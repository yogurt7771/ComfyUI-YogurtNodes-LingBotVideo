import importlib.util
import sys


def _load_plugin_root(repo_root):
    package_name = "_yogurt_lingbot_video_plugin_contract"
    spec = importlib.util.spec_from_file_location(
        package_name,
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name)


def _required_inputs(node_class):
    return node_class.INPUT_TYPES().get("required", {})


def test_root_package_exports_yogurt_prefixed_loader_generation_and_rewriter_nodes(
    repo_root, folder_paths_stub
):
    plugin = _load_plugin_root(repo_root)

    assert plugin.NODE_CLASS_MAPPINGS
    assert set(plugin.NODE_DISPLAY_NAME_MAPPINGS) == set(plugin.NODE_CLASS_MAPPINGS)
    assert all(name.startswith("Yogurt") for name in plugin.NODE_CLASS_MAPPINGS)

    nodes = tuple(plugin.NODE_CLASS_MAPPINGS.values())
    assert any(
        "model_name" in _required_inputs(node) and "download_model" in _required_inputs(node)
        for node in nodes
    )
    assert any(node.RETURN_TYPES == ("IMAGE",) for node in nodes)
    assert any("prompt" in _required_inputs(node) and "STRING" in node.RETURN_TYPES for node in nodes)


def test_download_model_inputs_are_explicit_and_default_to_false(repo_root, folder_paths_stub):
    plugin = _load_plugin_root(repo_root)
    inputs_with_download = [
        required["download_model"]
        for node in plugin.NODE_CLASS_MAPPINGS.values()
        if "download_model" in (required := _required_inputs(node))
    ]

    assert inputs_with_download
    assert all(spec[0] == "BOOLEAN" for spec in inputs_with_download)
    assert all(spec[1]["default"] is False for spec in inputs_with_download)


def test_nodes_expose_all_public_generation_modes(repo_root, folder_paths_stub):
    plugin = _load_plugin_root(repo_root)
    declared_mode_choices = []
    for node in plugin.NODE_CLASS_MAPPINGS.values():
        required = _required_inputs(node)
        if "mode" in required:
            declared_mode_choices.append(tuple(required["mode"][0]))

    assert declared_mode_choices
    assert all(set(choices) == {"t2i", "t2v", "ti2v"} for choices in declared_mode_choices)


def test_requirements_have_no_exact_pins_and_keep_transformers_below_v5(repo_root):
    requirements = [
        line.strip()
        for line in (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements
    assert all("==" not in requirement for requirement in requirements)
    transformers_requirement = next(
        requirement for requirement in requirements if requirement.lower().startswith("transformers")
    )
    assert "<5" in transformers_requirement.replace(" ", "")
