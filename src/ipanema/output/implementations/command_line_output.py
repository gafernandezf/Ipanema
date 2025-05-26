from venv import logger
from ipanema.model import ModelPlugin
from ipanema.output import OutputPlugin

class CommandLineOutput(OutputPlugin):
    """Abstraction of a generic output plugin"""

    def generate_results(self, model: ModelPlugin) -> None:
        """
        Prints results via command line for a fitted model.
        
        Assumes fit_manager property is a Minuit instance prepared for fitting.
        """

        logger.info(f"Starting Results Generation on '{self.__class__}'"
                    f" for model '{model.__class__}'")
        
        model.fit_manager.migrad()
        model.fit_manager.hesse()

        print(model.fit_manager.values)
        print(model.fit_manager.errors)

        logger.info(f"'{self.__class__}' Execution Complete")
