from abc import ABC, abstractmethod

class InputPlugin(ABC):

    @staticmethod
    @abstractmethod
    def getParams() -> dict:
        """Prepares data for a model initialization."""
        pass