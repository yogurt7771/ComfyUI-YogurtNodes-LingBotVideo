import importlib
import socket
import sys
import types
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("tests must not access the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def folder_paths_stub(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    registrations = []
    module = types.ModuleType("folder_paths")
    module.models_dir = str(model_dir)
    module.folder_names_and_paths = {}

    def add_model_folder_path(name, path, is_default=False):
        registrations.append((name, Path(path), is_default))
        existing = module.folder_names_and_paths.get(name, ([], set()))
        module.folder_names_and_paths[name] = ([*existing[0], str(path)], existing[1])

    def get_folder_paths(name):
        return list(module.folder_names_and_paths.get(name, ([], set()))[0])

    module.add_model_folder_path = add_model_folder_path
    module.get_folder_paths = get_folder_paths
    monkeypatch.setitem(sys.modules, "folder_paths", module)
    return types.SimpleNamespace(module=module, model_dir=model_dir, registrations=registrations)


@pytest.fixture
def fresh_import():
    def load(module_name):
        for name in list(sys.modules):
            if name == "yogurt_lingbot_video" or name.startswith("yogurt_lingbot_video."):
                sys.modules.pop(name)
        return importlib.import_module(module_name)

    yield load

    for name in list(sys.modules):
        if name == "yogurt_lingbot_video" or name.startswith("yogurt_lingbot_video."):
            sys.modules.pop(name)


@pytest.fixture
def repo_root():
    return REPOSITORY_ROOT
