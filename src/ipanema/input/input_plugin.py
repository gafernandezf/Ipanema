from abc import ABC, abstractmethod

class InputPlugin(ABC):

    @staticmethod
    @abstractmethod
    def get_params() -> dict:
        """Prepares data for a model initialization."""
        pass