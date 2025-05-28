from types import ModuleType
import pytest
from unittest import mock
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import importlib
from pathlib import Path
from ipanema.core import Core 
from ipanema.input.input_plugin import InputPlugin
from ipanema.model.model_plugin import ModelPlugin
from ipanema.output.output_plugin import OutputPlugin
from ipanema.exceptions import (
    IpanemaInitializationError, 
    IpanemaFittingError, 
    IpanemaOutputError, 
    IpanemaImportError
)


#####################
# _class_from_module
#####################


@pytest.mark.parametrize("input_str,expected", [
    ("default_plugin_name", "DefaultPluginName"),
    ("example", "Example"),
    ("plugin_2_test", "Plugin2Test"),
    ("a_b_c", "ABC"),
    ("", ""),
    ("__double__underscores__", "DoubleUnderscores"),
    ("___", "")
])
def test_valid_snake_case(input_str, expected):
    assert Core._class_from_module(input_str) == expected

def test_non_string_input():
    with pytest.raises(AttributeError):
        Core._class_from_module(None)


###################
# _retrieve_module
###################


def create_temp_module(
        directory: Path, 
        module_name: str, 
        content: str = "value = 42"
    ) -> Path:
    file_path = directory / f"{module_name}.py"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def temp_dir():
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path)

def test_load_module_from_custom_path(temp_dir):
    module_name = "testmodule"
    create_temp_module(temp_dir, module_name)

    module = Core._retrieve_module(
        custom_paths=[temp_dir],
        default_path="fake.default",
        module_name=module_name,
    )
    assert isinstance(module, ModuleType)
    assert hasattr(module, "value")
    assert module.value == 42

@patch("importlib.import_module")
def test_retrieve_module_default(mock_import):
    mock_module = MagicMock()
    mock_import.return_value = mock_module

    result = Core._retrieve_module(
        custom_paths=[],
        default_path="ipanema.model.implementations",
        module_name="default_model"
    )

    mock_import.assert_called_once_with(
        "ipanema.model.implementations.default_model"
    )
    assert result == mock_module

def test_module_not_found_anywhere_raises():
    with pytest.raises(ModuleNotFoundError):
        Core._retrieve_module(
            custom_paths=[],
            default_path="fake_package",
            module_name="non_existent"
        )

def test_invalid_module_in_custom_path_logs_error(temp_dir, caplog):
    module_name = "badmodule"
    # Create a broken module (syntax error)
    create_temp_module(temp_dir, module_name, content="def broken code")

    with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError):
        with pytest.raises(ModuleNotFoundError):
            Core._retrieve_module(
                custom_paths=[temp_dir],
                default_path="not.a.real.package",
                module_name=module_name,
            )
    assert any(
        "Problem searching module" in rec.message for rec in caplog.records
    )


###################
# _resolve_plugins
###################


class FakeInputPlugin:
    @staticmethod
    def get_params():
        return {}
    
class FakeModelPlugin:
    def __init__(self, params):
        self.params = params

    def prepare_fit(self):
        pass

class FakeOutputPlugin:
    def generate_results(self, model):
        pass


@pytest.fixture
def mock_plugin_modules(monkeypatch):
    """Mocks _retrieve_module and _class_from_module for all plugin types"""
    
    dummy_input_module = ModuleType("input_module")
    dummy_model_module = ModuleType("model_module")
    dummy_output_module1 = ModuleType("output_module1")
    dummy_output_module2 = ModuleType("output_module2")

    dummy_input_module.InputPlugin = FakeInputPlugin
    dummy_model_module.ModelPlugin = FakeModelPlugin
    dummy_output_module1.OutputPluginA = FakeOutputPlugin
    dummy_output_module2.OutputPluginB = FakeOutputPlugin

    # Mock _retrieve_module to return predefined modules
    retrieve_mock = mock.MagicMock(side_effect=[
        dummy_input_module,
        dummy_model_module,
        dummy_output_module1,
        dummy_output_module2,
    ])

    # Mock _class_from_module to return expected class names
    class_name_mock = mock.MagicMock(side_effect=[
        "InputPlugin", "ModelPlugin", "OutputPluginA", "OutputPluginB"
    ])

    monkeypatch.setattr("ipanema.core.Core._retrieve_module", retrieve_mock)
    monkeypatch.setattr("ipanema.core.Core._class_from_module", class_name_mock)

@pytest.fixture
def mock_config(monkeypatch):
    
    CONFIG_MOCK = {
        "custom_paths": [],
        "input": "input_plugin",
        "model": "model_plugin",
        "output": ["output_plugin_a", "output_plugin_b"]
    }

    monkeypatch.setattr("ipanema.config.config.CONFIG", CONFIG_MOCK)

    # Enums Mocks
    class FakeCore:
        class PluginPath:
            INPUT_PATH = mock.Mock(value="input")
            MODEL_PATH = mock.Mock(value="model")
            OUTPUT_PATH = mock.Mock(value="output")
        class PluginType:
            INPUT = mock.Mock(value="input")
            MODEL = mock.Mock(value="model")
            OUTPUT = mock.Mock(value="output")
        class DefaultPlugin:
            DEFAULT_INPUT = mock.Mock(value="default_input")
            DEFAULT_MODEL = mock.Mock(value="default_model")
            DEFAULT_OUTPUT = mock.Mock(value="default_output")

    monkeypatch.setattr("ipanema.core.Core", FakeCore)

def test_resolve_plugins_success(mock_plugin_modules, mock_config):
    plugin_loader = Core()
    input_class, model_class, output_classes = plugin_loader._resolve_plugins()

    assert input_class is FakeInputPlugin
    assert model_class is FakeModelPlugin
    assert output_classes == [FakeOutputPlugin, FakeOutputPlugin]

def test_resolve_plugins_fails(monkeypatch):
    plugin_loader = Core()

    monkeypatch.setattr(
        "ipanema.core.Core._retrieve_module", 
        mock.Mock(side_effect=ImportError("ImportError"))
    )
    with pytest.raises(IpanemaImportError) as exc_info:
        plugin_loader._resolve_plugins()

    assert "Problem during module import" in str(exc_info.value)


##############
# run_ipanema
##############


@pytest.fixture
def plugin_loader(monkeypatch):
    loader = Core()

    monkeypatch.setattr(
        loader, "_resolve_plugins",
        mock.Mock(return_value=(
            FakeInputPlugin, 
            FakeModelPlugin, 
            [FakeOutputPlugin]
        ))
    )
    return loader

def test_run_ipanema_import_error(monkeypatch):
    loader = Core()
    monkeypatch.setattr(
        loader, "_resolve_plugins",
        mock.Mock(side_effect=IpanemaImportError("Import fail"))
    )

    with mock.patch("venv.logger.critical") as mock_log:
        loader.run_ipanema()
        mock_log.assert_called_once()
        assert "Problem during module import" in mock_log.call_args[0][0]

def test_run_ipanema_input_error(plugin_loader):
    class FailingInput:
        @staticmethod
        def get_params():
            raise ValueError("Input Error")

    plugin_loader._resolve_plugins.return_value = (
        FailingInput, 
        FakeModelPlugin, 
        [FakeOutputPlugin]
    )

    with pytest.raises(IpanemaInitializationError):
        plugin_loader.run_ipanema()

def test_run_ipanema_model_error(plugin_loader):
    class FailingModel:
        def __init__(self, params):
            pass

        def prepare_fit(self):
            raise RuntimeError("Model Error")

    plugin_loader._resolve_plugins.return_value = (
        FakeInputPlugin, 
        FailingModel, 
        [FakeOutputPlugin]
    )

    with pytest.raises(IpanemaFittingError):
        plugin_loader.run_ipanema()

def test_run_ipanema_output_error(plugin_loader):
    class FailingOutput:
        def generate_results(self, model):
            raise Exception("Fail in output")

    plugin_loader._resolve_plugins.return_value = (
        FakeInputPlugin, 
        FakeModelPlugin, 
        [FailingOutput]
    )

    with pytest.raises(IpanemaOutputError):
        plugin_loader.run_ipanema()