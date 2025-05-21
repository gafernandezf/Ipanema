import pytest

from ipanema.output.output_plugin import OutputPlugin

def test_ModelPlugin_cannot_be_instantiated():
    with pytest.raises(TypeError):
        OutputPlugin()