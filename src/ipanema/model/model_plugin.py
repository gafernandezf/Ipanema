from abc import ABC, abstractmethod
from iminuit import Minuit

class ModelPlugin(ABC):
    """
    Abstraction of an Ipanema's Model Plugin.
    
    Type of Plugin dedicated to the preparation of an arbitrary model to 
    be fitted.

    Atributtes:
        fit_manager (Minuit): Function minimizer and error computer used during
            the fitting process
        parameters (dict): Dictionary containing the parameters required during 
            'fit_manager' initialization. 
    """

    _fit_manager: Minuit
    _parameters: dict

    def __init__(self, params: dict) -> None:
        self._parameters = params

    @abstractmethod
    def prepare_fit(self) -> None:
        """
        Prepares 'fit_manager' for its use.
        
        Initializes 'fit_manager' using the 'parameters' previously provided.
        """
        pass

    @property
    def fit_manager(self) -> Minuit:
        """Getter for fit_manager property."""
        return self._fit_manager
    
    @fit_manager.setter
    def fit_manager(self, manager: Minuit):
        """Setter for fit_manager property."""
        self._fit_manager = manager
    
    @property
    def parameters(self) -> dict:
        """Getter for parameters property."""
        return self._parameters