from abc import ABC, abstractmethod

class AbstractGeneticAlgorithm(ABC):
    
    @abstractmethod
    def real_selection():
        pass

    @abstractmethod
    def complex_selection():
        pass

    @abstractmethod
    def real_mutation():
        pass

    @abstractmethod
    def complex_mutation():
        pass