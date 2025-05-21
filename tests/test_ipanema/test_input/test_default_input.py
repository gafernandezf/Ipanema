import pytest

from ipanema.input.implementations.default_input import DefaultInput

def test_get_params_exists():
    assert hasattr(DefaultInput, "get_params")
    assert callable(DefaultInput.get_params)

def test_get_params_returns_dict():
    params = DefaultInput.get_params()
    assert isinstance(params, dict)

def test_get_params_returns_empty_dict():
    params = DefaultInput.get_params()
    assert params == {}