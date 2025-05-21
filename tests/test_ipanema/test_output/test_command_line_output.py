import pytest
from unittest.mock import MagicMock
from iminuit import Minuit
from ipanema.model.model_plugin import ModelPlugin
from ipanema.output.implementations.command_line_output import CommandLineOutput

def test_generate_results_calls_fit_manager_methods():

    fit_manager_mock = MagicMock(Minuit)
    model_plugin_mock = MagicMock(ModelPlugin)
    model_plugin_mock.fit_manager = fit_manager_mock

    output = CommandLineOutput()
    output.generate_results(model_plugin_mock)

    fit_manager_mock.migrad.assert_called_once()
    fit_manager_mock.hesse.assert_called_once()