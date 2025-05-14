from ipanema.model import ModelPlugin
from ipanema.output import OutputPlugin

class CommandLineOutput(OutputPlugin):
    """Abstraction of a generic output plugin"""

    def generate_results(model: ModelPlugin) -> None:
        """
        Prints results via command line for a fitted model.
        
        Assumes fit_manager property is a Minuit instance prepared for fitting.
        """

        model.fit_manager.migrad()
        model.fit_manager.hesse()