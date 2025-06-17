from ipanema.model import ModelPlugin
from ipanema.output import OutputPlugin

class CommandLineOutput(OutputPlugin):
    """
    Output plugin that performs the fit and displays results via the 
    command-line interface (CLI).

    This plugin uses the model's fit manager to run the minimization and 
    then prints the parameter values and errors to the console.
    """

    def generate_results(self, model: ModelPlugin) -> None:
        """
        Fit the model and print the results to the command line.

        This method runs the fit using the model's fit manager (assumed to be 
        a Minuit instance), then prints the optimized parameter values and 
        their estimated errors.

        Args:
            model (ModelPlugin): The model to be fitted and whose results 
                will be displayed.
        """
        
        model.fit_manager.migrad()
        model.fit_manager.hesse()

        print(f"\nFit Manager Values: \n{model.fit_manager.values}\n")
        print(f"\nFit Manager Error: \n{model.fit_manager.errors}\n")
