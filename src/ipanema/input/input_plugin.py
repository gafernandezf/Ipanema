from abc import ABC, abstractmethod

class InputPlugin(ABC):
    """
    Abstraction base class for Ipanema's Input Plugin.
    
    This type of plugin is responsible for processing and parsing parameters 
    required for an arbitrary simulation.
    """

    @staticmethod
    @abstractmethod
    def get_params() -> dict:
        """
        Prepares data for a model initialization.

        Returns:
            dict: A dictionary containing the parameters required by a 
                Model Plugin.
        """
        pass