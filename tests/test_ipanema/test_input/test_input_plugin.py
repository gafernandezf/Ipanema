import pytest

from ipanema.input.input_plugin import InputPlugin

def test_ModelPlugin_cannot_be_instantiated():
    with pytest.raises(TypeError):
        InputPlugin()