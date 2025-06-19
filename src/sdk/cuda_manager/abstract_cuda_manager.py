from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from sdk.cuda_manager.cuda_program import  CudaProgram

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
                int, tuple[tuple[int, ...], Any]
            ],
            block: tuple[int, int, int],
            grid: tuple[int, int],
            *args
    ) -> list:
        """
        Executes a registered CUDA kernel with the given arguments and 
        returns specified output buffers.

        This method runs a compiled CUDA function by name, sending all 
        arguments to the GPU, and returning a list of output buffers as host 
        copies based on 'outputs_idx' and 'outputs_details'.

        Args:
            func_name (str): Name of the CUDA function to execute.
            output_idx (list[int]): Indices in 'args' corresponding to output 
                buffers.
            output_details (dict[int, tuple[tuple[int, ...], Any]): Formal
                description of each argument following the structure
                'output_idx : (shape, dtype)'.
            block (tuple[int, int, int], optional): CUDA block dimensions. 
                Defaults to (256,1,1).
            grid (tuple[int, int], optional): CUDA grid dimensions. 
                Defaults to (1,1).
            *args: Parameters for the CUDA function.

        Returns:
            list: List with each one of the outputs from the CUDA function.

        Example:
            >>> outputs = cuda_manager.run_program(
            ...     "my_kernel",
            ...     [1],
            ...     {1: ((100,), np.float64)},
            ...     block=(256,1,1),
            ...     grid=(10,1),
            ...     input_array, np.empty(100), 42
            ... )
            >>> print(outputs[0].shape)
            (100,)
        """
        pass

    @abstractmethod
    def single_operation(self, func_name: str, *args) -> Any:
        """
        Performs a simple operation using CUDA.
        
        Provides access to GPU-accelerated functions defined by a specific CUDA
        library. Handles any GPU-compatible processing needed for its execution. 

        Args:
            func_name (str): Name of the desired CUDA operation implemented 
                by a specific CUDA library.
            *args: List of arguments needed for the desired operation that will
                be processed for GPU-compatibility.

        Returns:
            Any: A host copy of the result of the CUDA operation

        Raises:
            AttributeError: If 'func_name' does not exist in this specific CUDA
                library.
        """
        pass

    @abstractmethod
    def reduction_operation(self, op_name: str, array: Any) -> Any:
        """
        Performs reduction operation for an array using a specific CUDA library.

        Provides access to reduction operations (sum, max, min, etc.) defined 
        by a specific CUDA library. Handles any GPU-compatible processing 
        needed for its execution. 

        Args:
            op_name (str): Name of the desired reduction operation implemented 
                by the specific CUDA library.
            array (Any): Data array to be reduced.

        Returns:
            Any: A host copy of the result of the reduction operation.

        Raises:
            AttributeError: If 'op_name' does not exist in the CUDA library.
        """
        pass

    def add_code_fragment(self, name: str, function: str | Path) -> None:
        """
        Registers a new CUDA code fragment by name.

        Args:
            name (str): Identifier for the code fragment.
            function (str | Path): CUDA source code string or path 
                to the file.
        """
        self.__src_code[name] = CudaProgram(function)

    def pop_code_fragment(self, name: str) -> str:
        """
        Removes and returns a previously registered CUDA code fragment.

        Args:
            name (str): Identifier of the code fragment to remove.

        Returns:
            str: The combined CUDA source code for the removed fragment.
        """
        program = self.__src_code.pop(name)
        program.includes.append(program.functions)
        return "\n".join(program.includes)

    # Getter for src_code
    @property
    def src_code(self) -> dict[str, CudaProgram]:
        """Getter for src_code property."""
        return self.__src_code
