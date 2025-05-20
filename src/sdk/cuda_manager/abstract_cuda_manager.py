from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple
from sdk.cuda_manager import  CudaProgram

class CudaManager(ABC):
    """Abstraction of a generic CUDA handler for Python."""

    __src_code: dict[str, CudaProgram]

    def __init__(self)-> None:
        """Initializes a CUDA program handler."""
        self.__src_code: dict[str, CudaProgram] = {}

    @abstractmethod
    def run_program(self,
            func_name: str,
            outputs_idx: list[int],
            outputs_details: dict[
                int, Tuple[Tuple[int, ...], Any]
            ],
            block: Tuple[int, int, int],
            grid: Tuple[int, int],
            *args
    ) -> list:
        """Calls a specific CUDA function."""
        pass

    @abstractmethod
    def single_operation(self, func_name: str, *args) -> Any:
        """Performs a simple operation using CUDA."""
        pass

    @abstractmethod
    def reduction_operation(self, op_name: str, array: Any) -> Any:
        """Performs a reduction operation for a specific array"""
        pass

    def add_code_fragment(self, name: str, function: str | Path) -> None:
        """Adds a given global CUDA function."""
        self.__src_code[name] = CudaProgram(function)

    def pop_code_fragment(self, name: str) -> str:
        """Pops a specific fragment of the program as a string."""
        program = self.__src_code.pop(name)
        program.includes.append(program.functions)
        return "\n".join(program.includes)

    # Getter for src_code
    @property
    def src_code(self) -> dict[str, CudaProgram]:
        """Getter for src_code property."""
        return self.__src_code
