from ipanema.model import ModelPlugin
from ipanema.output import OutputPlugin

class CommandLineOutput(OutputPlugin):
    """Output Plugin for a CLI results displaying"""

    def generate_results(self, model: ModelPlugin) -> None:
        """
        Prints results via command line for a fitted model.
        
        Assumes fit_manager property is a Minuit instance prepared for fitting.
        """
        
        model.fit_manager.migrad()
        model.fit_manager.hesse()

        print(f"\nFit Manager Values: \n{model.fit_manager.values}\n")
        print(f"\nFit Manager Error: \n{model.fit_manager.errors}\n")
