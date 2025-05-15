from abc import ABC, abstractmethod
from pathlib import Path
from sdk.cuda_manager.cuda_function.cuda_function import CudaFunction

class CudaManager(ABC):
    """Abstraction of a generic CUDA handler for Python."""

    # CUDA parameters
    _num_block: int
    _num_threads_block: int
    _shared_mem_size: int

    # CUDA functions (kernels)
    __global_functions: dict[str, CudaFunction]
    __device_functions: dict[str, CudaFunction]


    def __init__(
            self,
            num_blocks: int,
            num_threads_block: int,
            shared_mem_size: int = None,
    )-> None:
        """Initializes a CUDA program handler."""
        self._num_blocks = num_blocks
        self._num_threads_block = num_threads_block
        if shared_mem_size is None:
            self._shared_mem_size = -1
        else:
            self._shared_mem_size = shared_mem_size
        self._global_functions: dict[str, CudaFunction] = {}
        self._device_functions: dict[str, CudaFunction] = {}


    def add_global_function(self, name: str, function: str | Path) -> None:
        """Adds a given global CUDA function."""
        self._global_functions[name] = CudaFunction(function)

    def add_device_function(self, name: str, function: str | Path) -> None:
        """Adds a given device CUDA function."""
        self._device_functions[name] = CudaFunction(function)

    def pop_global_function(self, name: str) -> str:
        """Pops a specific global function as a string."""
        return self._global_functions.pop(name).function

    def pop_device_function(self, name: str) -> str:
        """Pops a specific device function as a string."""
        return self._device_functions.pop(name).function

    @abstractmethod
    def run_program(self, func_name: str, args, outputs) -> any:
        """Runs a specific sequence of CUDA functions."""
        pass   


    # Getter for num_block
    @property
    def num_block(self) -> int:
        """Getter for num_block property"""
        return self._num_block

    # Setter for num_block
    @num_block.setter
    def num_block(self, num_block: int) -> None:
        """Setter for num_block property"""
        if isinstance(num_block, int):
            self._num_block = num_block
        else:
            raise TypeError(
                f"Type {type(num_block)} not admitted for num_block"
            )
        
    # Getter for num_threads_block
    @property
    def num_threads_block(self) -> int:
        """Getter for num_threads_block property"""
        return self._num_threads_block

    # Setter for num_threads_block
    @num_threads_block.setter
    def num_threads_block(self, num_threads_block: int) -> None:
        """Setter for num_threads_block property"""
        if isinstance(num_threads_block, int):
            self._num_threads_block = num_threads_block
        else:
            raise TypeError(
                f"Type {type(num_threads_block)} \
                not admitted for num_threads_block"
            )
    
    # Getter for shared_mem_size
    @property
    def shared_mem_size(self) -> int:
        """Getter for shared_mem_size property"""
        return self._shared_mem_size

    # Setter for shared_mem_size
    @shared_mem_size.setter
    def shared_mem_size(self, shared_mem_size: int) -> None:
        """Setter for shared_mem_size property"""
        if isinstance(shared_mem_size, int):
            self._shared_mem_size = shared_mem_size
        else:
            raise TypeError(
                f"Type {type(shared_mem_size)} \
                not admitted for shared_mem_size"
            )
        
    # Getter for global_functions
    @property
    def global_functions(self) -> dict[str, CudaFunction]:
        """Getter for global_functions property"""
        return self.__global_functions
    
    # Getter for device_functions
    @property
    def device_functions(self) -> dict[str, CudaFunction]:
        """Getter for device_functions property"""
        return self.__device_functions
        