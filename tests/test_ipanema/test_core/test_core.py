from unittest.mock import MagicMock, patch
from pathlib import Path

from ipanema.core import Core

def test_class_from_module():
    assert Core._class_from_module("default_input") == "DefaultInput"
    assert Core._class_from_module("default_model") == "DefaultModel"
    assert Core._class_from_module("command_line_output") == "CommandLineOutput"

@patch("importlib.import_module")
def test_retrieve_module_default(mock_import):
    mock_module = MagicMock()
    mock_import.return_value = mock_module

    result = Core._retrieve_module(
        custom_paths=[],
        default_path="ipanema.model.implementations",
        module_name="default_model"
    )

    mock_import.assert_called_once_with("ipanema.model.implementations.default_model")
    assert result == mock_module

# TODO test para run_ipanema()