import importlib

from ipanema.config.config import CONFIG
from ipanema.input import InputPlugin
from ipanema.model import ModelPlugin
from ipanema.output import OutputPlugin

# NamedTuple

class Core():

    @staticmethod
    def run_ipanema(self) -> None:
        """Ipanema execution using 'ipanema.config'."""

        # Path definition for the input
        input_path = "ipanema.input.implementations"
        input_file_name = CONFIG.get(
            "input",
            "default_input"
        )
        input_path = ".".join([input_path, input_file_name])

        # Path definition for the model
        model_path = "ipanema.model.implementations"
        model_file_name = CONFIG.get(
            "model",
            "default_model"
        )
        model_path = ".".join([model_path, model_file_name])

        # Path definition for the outputs
        output_path = "ipanema.output.implementations"
        output_file_names = CONFIG.get(
            "output", 
            ["command_line_output"]
        )
        output_paths = [
            ".".join([output_path, o]) for o in output_file_names
        ]
        
        # Input Module import
        input_module = importlib.import_module(input_path)
        # Model Module import
        model_module = importlib.import_module(model_path)
        # Output Modules import
        output_modules = [
            importlib.import_module(o) for o in output_paths
        ]

        # Input Class import
        InputClass = getattr(
            input_module,
            self._class_from_file(input_file_name)
        )
        # Model Class import
        ModelClass = getattr(
            model_module,
            self._class_from_file(model_file_name)
        )
        # Output Classes import
        output_classes = [
            getattr(o_module, self._class_from_file(o_file_name))
            for o_module, o_file_name in zip(output_modules, output_file_names)
        ]

        # Preparing the model fit with the parameters
        model: ModelPlugin = ModelClass(InputClass.getParams())
        model.prepare_fit()
        # Model fit and Results presentation
        outputs: list[OutputPlugin] = [
            OutputClass() for OutputClass in output_classes
        ]
        for output in outputs:
            output.generate_results(model)

    @staticmethod
    def _class_from_file(file_name: str) -> str:
        """Parses the file name (without file extension) to class name."""
        return "".join(token.capitalize() for token in file_name.split("_"))
    