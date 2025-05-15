from functools import singledispatchmethod
from pathlib import Path

class CudaFunction():
    """Entity representing a CUDA kernel."""

    __function: str

    @singledispatchmethod
    def __init__(self, function: str | Path) -> None:
        raise TypeError(
            f"Type {type(function)} not admitted for a function"
        )

    @__init__.register(str)
    def _(self, function: str) -> None:
        """Initializes a cuda function using the function itself a string"""
        self.__function: str = function

    @__init__.register(Path)
    def _(self, function: Path) -> None:
        """Initializes a cuda function using the file path of the function"""
        with open(function, 'r', encoding='UTF-8') as file:
            self.__function: str = file.read()

    @property
    def function(self) -> str:
        """Getter for function property"""
        return self.__function

    @function.setter
    def function(self, func: str | Path) -> None:
        """Setter for function property"""
        if isinstance(func, str):
            self.__function: str = func
        elif isinstance(func, Path):
            with open(function, 'r', encoding='UTF-8') as file:
                self.__function: str = file.read()
        else:
            raise TypeError(
                f"Type {type(func)} not admitted for a function"
            )
