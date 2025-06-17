from functools import singledispatchmethod
from pathlib import Path
from re import compile

class CudaProgram():
    """
    Entity which parses and represents a CUDA source code.
    
    This class abstracts a CUDA program by separating its '#include' 
    instructions from the body of the code. Allows initialization using the
    source code as a string or the file path of the program.

    Attributes:
        functions (str): Body of the CUDA program without include directives.
        includes (list[str]): List of include directives of the CUDA program.
    """

    __functions: str
    __includes: list[str]

    @singledispatchmethod
    def __init__(self, function) -> None:
        """
        Base constructor for CudaProgram (singledispatch).

        Raises:
            TypeError: If the provided input type is not supported.
        """
        raise TypeError(
            f"Type {type(function)} not admitted for a function"
        )

    @__init__.register(str)
    def _(self, function: str) -> None:
        self.__save_src_code(function)

    @__init__.register(Path)
    def _(self, function: Path) -> None:
        with open(function, 'r', encoding='UTF-8') as file:
            self.__save_src_code(file.read())

    def __save_src_code(self, src_code: str) -> None:
        """
        Parses the CUDA source code and separates the include directives
        from the main function body.

        Arguments:
            src_code (str): Complete CUDA program as a string.
        """
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
        """Getter for functions property"""
        return self.__functions

    @property
    def includes(self) -> list[str]:
        """Getter for includes property"""
        return self.__includes