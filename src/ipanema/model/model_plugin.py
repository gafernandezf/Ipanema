from abc import ABC, abstractmethod
from iminuit import Minuit

class ModelPlugin(ABC):
    """
    Abstract base class for Ipanema's Model Plugin.
    
    This type of plugin is responsible for preparing an arbitrary model 
    to be fitted.

    Atributtes:
        fit_manager (Minuit): Function minimizer and error estimator used during
            the fitting process.
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
        Prepare 'fit_manager' for use.
        
        Initializes 'fit_manager' using the previously provided parameters.
        """
        pass

    @property
    def fit_manager(self) -> Minuit:
        """Get the fit_manager."""
        return self._fit_manager
    
    @fit_manager.setter
    def fit_manager(self, manager: Minuit):
        """Set the fit_manager."""
        self._fit_manager = manager
    
    @property
    def parameters(self) -> dict:
        """Get the parameters dictionary."""
        return self._parameters