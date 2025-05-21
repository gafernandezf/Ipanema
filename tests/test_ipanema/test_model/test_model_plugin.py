import pytest
from ipanema.model.model_plugin import ModelPlugin
from types import SimpleNamespace

class BasicModel(ModelPlugin):

    def prepare_fit(self):
        pass

def test_ModelPlugin_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ModelPlugin({"x": -1})

def test_modelplugin_initialization():
    params = {"x": -1, "y": 0}
    model = BasicModel(params)
    assert model.parameters == params

def test_fit_manager_set_get():
    model = BasicModel({"x": -1})
    test_manager = SimpleNamespace()
    model.fit_manager = test_manager
    assert model.fit_manager == test_manager