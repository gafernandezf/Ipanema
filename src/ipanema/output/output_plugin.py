from abc import ABC, abstractmethod
from ipanema.model import ModelPlugin

class OutputPlugin(ABC):
    """
    Abstract base class for Ipanema's Output Plugin.
    
    This type of plugin is responsible for managing model fitting and 
    presenting results.
    """

    @abstractmethod
    def generate_results(self, model: ModelPlugin) -> None:
        """
        Generate and present results for a fitted model.

        Arguments:
            model (ModelPlugin): The fitted model to process results from.
        """
        pass