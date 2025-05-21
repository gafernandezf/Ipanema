from abc import ABC, abstractmethod
from iminuit import Minuit

class ModelPlugin(ABC):
    """Abstraction of a generic model plugin"""

    _fit_manager: Minuit
    _parameters: dict

    def __init__(self, params: dict) -> None:
        """Initializes the model."""
        self._parameters = params

    @abstractmethod
    def prepare_fit(self) -> None:
        """Fits this model using parameters provided during initialization."""
        pass

    # Getter for fit_manager
    @property
    def fit_manager(self) -> Minuit:
        """Getter for fit_manager property"""
        return self._fit_manager
    
    # Setter for fit_manager
    @fit_manager.setter
    def fit_manager(self, manager: Minuit):
        self._fit_manager = manager
    
    # Getter for parameters
    @property
    def parameters(self) -> dict:
        """Getter for parameters property"""
        return self._parameters