from abc import ABC, abstractmethod
# TODO import Minuit

class ModelPlugin(ABC):
    """Abstraction of a generic model plugin"""

    _fit_manager: Minuit

     # Getter for fit_manager
    @property
    def fit_manager(self) -> Minuit:
        """Getter for fit_manager property"""
        return self._fit_manager

    @abstractmethod
    def __init__(self, params: dict) -> None:
        """Initializes the model."""
        pass

    @abstractmethod
    def prepare_fit() -> None:
        """Fits this model using parameters provided during initialization."""
        pass