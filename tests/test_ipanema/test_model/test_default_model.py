from iminuit import Minuit
from ipanema.model.implementations.default_model import DefaultModel

def test_init():
    params = {}
    model = DefaultModel(params)
    assert hasattr(model, "parameters")
    assert model.parameters == params

def test_generate_fcn():
    model = DefaultModel({})
    fcn = model._generate_fcn()
    assert fcn(3) == 0
    assert fcn(0) == 9

def test_prepare_fit_creates_minuit():
    model = DefaultModel({})
    model.prepare_fit()
    assert isinstance(model.fit_manager, Minuit)
    assert "x" in model.fit_manager.parameters
    assert model.fit_manager.values["x"] == 1