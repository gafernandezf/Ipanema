from unittest.mock import MagicMock
from pathlib import Path

from ipanema.core import Core

def test_class_from_module():
    assert Core._class_from_module("default_input") == "DefaultInput"
    assert Core._class_from_module("default_model") == "DefaultModel"
    assert Core._class_from_module("command_line_output") == "CommandLineOutput"

