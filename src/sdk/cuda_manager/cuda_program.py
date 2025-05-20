from functools import singledispatchmethod
from pathlib import Path
from re import compile

class CudaProgram():
    """Entity representing a CUDA source code."""

    __functions: str
    __includes: list[str]

    @singledispatchmethod
    def __init__(self, function) -> None:
        raise TypeError(
            f"Type {type(function)} not admitted for a function"
        )

    @__init__.register(str)
    def _(self, function: str) -> None:
        """Initializes a cuda function using the function itself a string"""
        self.__save_src_code(function)

    @__init__.register(Path)
    def _(self, function: Path) -> None:
        """Initializes a cuda function using the file path of the function"""
        with open(function, 'r', encoding='UTF-8') as file:
            self.__save_src_code(file.read())

    def __save_src_code(self, src_code: str) -> None:

        include_list: list[str] = []
        function_list: list[str] = []
        include_pattern: str = r"^[ \t]*#include\s"

        expr = compile(include_pattern)

        code_lines = src_code.splitlines()
        for line in code_lines:
            if expr.match(line):
                include_list.append(line)
            else:
                function_list.append(line)  
        self.__includes = include_list
        self.__functions = "\n".join(function_list)

    @property
    def functions(self) -> str:
        """Getter for function property"""
        return self.__functions

    @property
    def includes(self) -> list[str]:
        """Getter for includes property"""
        return self.__includes