from abc import ABC, abstractmethod

from ipanema.model import ModelPlugin

class OutputPlugin(ABC):
    """Abstraction of a generic output plugin"""

    @abstractmethod
    def generate_results(model: ModelPlugin) -> None:
        """Provides results for a fitted model in a specific output format."""
        pass