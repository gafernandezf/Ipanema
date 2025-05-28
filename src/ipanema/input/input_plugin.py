from abc import ABC, abstractmethod

class InputPlugin(ABC):
    """
    Abstraction of an Ipanema's Input Plugin.
    
    Type of Plugin dedicated to the parameter processing and parsing for an
    arbitrary simulation.  
    """

    @staticmethod
    @abstractmethod
    def get_params() -> dict:
        """
        Prepares data for a model initialization.
        
        Defines a dictionary containing the parameters needed by an arbitrary
        Model Plugin.

        Returns:
            dict: Dictionary formed by the expected parameters.
        """
        pass