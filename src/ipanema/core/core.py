from enum import Enum
import importlib
import importlib.util
import logging
import time
from pathlib import Path
from types import ModuleType
from venv import logger
from ipanema.config.config import CONFIG
from ipanema.exceptions import (
    IpanemaImportError, 
    IpanemaInitializationError, 
    IpanemaFittingError, 
    IpanemaOutputError
)
from ipanema.input import InputPlugin
from ipanema.model import ModelPlugin
from ipanema.output import OutputPlugin

# Namedtuple

class Core():
    """
    Main handler class for Ipanema's Plugin Based System.

    Attributes:
        PluginType (Enum): Enum for identifying plugin types 
            (input, model, output).
        PluginPath (Enum): Enum for defining base module paths for each 
            plugin type.
        DefaultPlugin (Enum): Enum for defining default plugin names.
    """

    class PluginType(Enum):
        INPUT: str = "input"
        MODEL: str = "model"
        OUTPUT: str = "outputs"

    class PluginPath(Enum):
        INPUT_PATH: str = "ipanema.input.implementations"
        MODEL_PATH: str = "ipanema.model.implementations"
        OUTPUT_PATH: str = "ipanema.output.implementations"

    class DefaultPlugin(Enum):
        DEFAULT_INPUT: str = "default_input"
        DEFAULT_MODEL: str = "default_model"
        DEFAULT_OUTPUT: str = "command_line_output"

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        ) 

    def run_ipanema(self) -> None:
        """
        Executes the complete Ipanema workflow.
 
        Prepares the model implemented in the model plugin using the parameters
        defined in the input plugin. Finally, runs the model and generates the
        results specified by the output plugins.

        Raises:
            IpanemaInitializationError: If Problems encountered during 
                input plugin execution.
            IpanemaFittingError: If Problems encountered during 
                model plugin execution.
            IpanemaOutputError: If Problems encountered during 
                output plugins execution.
        """
        try:
            InputClass, ModelClass, output_classes = self._resolve_plugins()
        except IpanemaImportError as e:
            logger.critical(f"Problem during module import" 
                            f"or class resolution: {e}")
            return
        
        logger.info(f"Obtaining parameters from '{InputClass}'")
        try:
            parameters: dict = InputClass.get_params()
        except Exception as e:
            logger.exception(
                f"Problem obtaining input parameters" 
                f" from '{InputClass}'"
            )
            raise IpanemaInitializationError(
                "Problem obtaining input parameters",
                e
            ) from e

        logger.info(f"Preparing model '{ModelClass}' for fitting")
        try:
            model: ModelPlugin = ModelClass(parameters)
            start_time: float = time.time()
            model.prepare_fit()
        except Exception as e:
            logger.exception(
                f"Problem during fit manager preparation" 
                f" in '{ModelClass}'"
            )
            raise IpanemaFittingError(
                "Problem during fit manager preparation", 
                e
            ) from e
        
        for OutputClass in output_classes:
            try:
                output: OutputPlugin = OutputClass()
                logger.info(
                    f"Starting Results Generation on '{OutputClass}'"
                    f" for model '{ModelClass}'"
                )
                output.generate_results(model)
            except Exception as e:
                logger.exception(
                    f"Problem during fit manager execution or results " 
                    f"presentation in '{OutputClass}'"
                )
                raise IpanemaOutputError(
                    "Problem during fit manager execution or results",
                    e
                ) from e
            
        end_time: float = time.time()
        logger.info(f"Calculation time: {end_time - start_time:.10f} seconds")
        
    def _resolve_plugins(
            self
        ) -> tuple[
            type[InputPlugin], 
            type[ModelPlugin], 
            list[type[OutputPlugin]]
        ]:
        """
        Dynamically loads the input, model and output plugins specified 
        in 'ipanema.config'.

        Raises:
            IpanemaImportError: If problems encountered during module importing
                or class resolution.
        """
        
        try:
            custom_paths: list[Path] = [
                Path(path) for path in CONFIG.get("custom_paths", [])
            ]

            # Path definition for the input
            input_path: str = Core.PluginPath.INPUT_PATH.value
            input_file_name: str = CONFIG.get(
                Core.PluginType.INPUT.value,
                Core.DefaultPlugin.DEFAULT_INPUT.value
            )
            if not input_file_name.strip():
                input_file_name = Core.DefaultPlugin.DEFAULT_INPUT.value

            # Path definition for the model
            model_path: str = Core.PluginPath.MODEL_PATH.value
            model_file_name: str = CONFIG.get(
                Core.PluginType.MODEL.value,
                Core.DefaultPlugin.DEFAULT_MODEL.value
            )
            if not model_file_name.strip():
                model_file_name = Core.DefaultPlugin.DEFAULT_MODEL.value

            # Path definition for the outputs
            output_path: str = Core.PluginPath.OUTPUT_PATH.value
            output_file_names: list[str] = CONFIG.get(
                Core.PluginType.OUTPUT.value, 
                [Core.DefaultPlugin.DEFAULT_OUTPUT.value]
            )
            output_file_names = [
                output for output in output_file_names if output.strip()
            ]
            if not output_file_names:
                output_file_names = [Core.DefaultPlugin.DEFAULT_OUTPUT.value]
            
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
                self._class_from_module(input_file_name)
            )
            # Model Class import
            ModelClass = getattr(
                model_module,
                self._class_from_module(model_file_name)
            )
            # Output Classes import
            output_classes = [
                getattr(o_module, self._class_from_module(o_file_name))
                for o_module, o_file_name in zip(
                    output_modules, 
                    output_file_names
                )
            ]
        except Exception as e:
            raise IpanemaImportError(f"Problem during module import or "
                                     f"class resolution: {e}") from e
        return InputClass, ModelClass, output_classes

    @staticmethod
    def _retrieve_module(
            custom_paths: list[Path], 
            default_path: str,
            module_name: str,
        ) -> ModuleType:
        """
        Loads dinamically a Python module given a list of custom file paths,
        importing from 'default_path' if not found elsewhere.

        Searches for a '.py' file matching 'module_name' in each path of 
        'custom_paths'. If the specified module is not found in any of the
        custom file paths it tries to import it from 'default_path' using
        'importlib.import_module'.

        Arguments:
            custom_paths (list[Path]): List of custom file paths.
            default_path (str): Standard dot-separated Python package path.
            module_name (str): Name of the '.py' file to search.

        Returns:
            ModuleType: A Python module imported.

        Raises:
            Exception: Possible exceptions encountered during module importing.
        """
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
                    logger.exception(f"Problem searching module '{module_name}'"
                                     f" at path '{path}': {e}")
        default_module = ".".join([default_path, module_name])
        try:
            return importlib.import_module(default_module)
        except Exception as e:
            logger.critical(f"Error importing default module "
                            f"'{default_module}': {e}")
            raise e

    @staticmethod
    def _class_from_module(file_name: str) -> str:
        """
        Parses a snake_case file name (without file extension) into a 
        PascalCase class name.

        Arguments:
            file_name (str): Name of a file without the file extension 
                in snake_case format.
        Returns:
            str: A class name in PascalCase format.

        Example:
            >>> class_name = _class_from_module("my_plugin_name")
            >>> print(class_name)
            MyPluginName

        """
        return "".join(token.capitalize() for token in file_name.split("_"))
    