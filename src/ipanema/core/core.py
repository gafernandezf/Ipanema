import importlib
import importlib.util

from pathlib import Path
from types import ModuleType
from venv import logger

from ipanema.config.config import CONFIG
from ipanema.input import InputPlugin
from ipanema.model import ModelPlugin
from ipanema.output import OutputPlugin

# NamedTuple

class Core():

    def run_ipanema(self) -> None:
        """Ipanema execution using 'ipanema.config'."""

        custom_paths: list[Path] = [
            Path(path) for path in CONFIG.get("custom_paths", [])
        ]

        # Path definition for the input
        input_path = "ipanema.input.implementations"
        input_file_name = CONFIG.get(
            "input",
            "default_input"
        )

        # Path definition for the model
        model_path = "ipanema.model.implementations"
        model_file_name = CONFIG.get(
            "model",
            "default_model"
        )

        # Path definition for the outputs
        output_path = "ipanema.output.implementations"
        output_file_names = CONFIG.get(
            "outputs", 
            ["command_line_output"]
        )
        
        # Input Module import
        input_module = self._retrieve_module(
            custom_paths, 
            input_path, 
            input_file_name
        )
        # Model Module import
        model_module = self._retrieve_module(
            custom_paths, 
            model_path, 
            model_file_name
        )
        # Output Modules import
        output_modules = [
            self._retrieve_module(
                custom_paths, 
                output_path, 
                output_file_name
            ) for output_file_name in output_file_names
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
        model: ModelPlugin = ModelClass(InputClass.get_params())
        model.prepare_fit()
        # Model fit and Results presentation
        outputs: list[OutputPlugin] = [
            OutputClass() for OutputClass in output_classes
        ]
        for output in outputs:
            output.generate_results(model)
        
    @staticmethod
    def _retrieve_module(
            custom_paths: list[Path], 
            default_path: str,
            module_name: str,
        ) -> ModuleType:

        for path in custom_paths:
            if path.exists():
                try:
                    file_name = ".".join([module_name, "py"])
                    spec = importlib.util.spec_from_file_location(
                        module_name, 
                        path.joinpath(file_name)
                    )
                    if spec:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module
                except (FileNotFoundError) as e:
                    pass # File not in this custom path
                except Exception as e:
                    logger.error(f"Problem searching module '{module_name}' \
                                 at path '{path}': {e}")
        default_module = ".".join([default_path, module_name])
        try:
            return importlib.import_module(default_module)
        except Exception as e:
            logger.critical(f"Error importing default module \
                                '{default_module}': {e}")
            raise e

    @staticmethod
    def _class_from_file(file_name: str) -> str:
        """Parses the file name (without file extension) to class name."""
        return "".join(token.capitalize() for token in file_name.split("_"))
    