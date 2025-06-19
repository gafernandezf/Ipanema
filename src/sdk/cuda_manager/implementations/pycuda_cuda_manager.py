from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Any
import numpy as np
import pycuda.cumath
import pycuda.gpuarray
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from sdk.cuda_manager.abstract_cuda_manager import CudaManager


class PyCudaManager(CudaManager, ABC):
    """Cuda Handler for PyCuda."""

    @abstractmethod
    def _initialize_context(self) -> None:
        """Initializes CUDA context."""
        pass

    @abstractmethod
    def _finish_up_context(self) -> None:
        """Cleans the CUDA context."""
        pass

    def run_program(self,
            func_name: str,
            outputs_idx: list[int],
            outputs_details: dict[
                int, tuple[tuple[int, ...], Any]
            ],
            block: tuple[int, int, int] = (256,1,1),
            grid: tuple[int, int] = (1,1),
            *args
    ) -> list:
        """
        Executes a registered CUDA kernel with the given arguments and 
        returns specified output buffers.

        This method runs a compiled CUDA function by name, sending all 
        arguments to the GPU, and returning a list of output buffers as host 
        copies based on 'outputs_idx' and 'outputs_details'.

        Args:
            func_name (str): Name of the called function.
            outputs_idx (list[int]): Indices of the arguments that are 
                output buffers.
            outputs_details (dict[int, tuple[tuple[int, ...], Any]): Formal
                description of each argument following the structure
                'output_idx : (shape, dtype)'.
            block (tuple[int, int, int], optional): CUDA block dimensions. 
                Defaults to (256,1,1).
            grid (tuple[int, int], optional): CUDA grid dimensions. 
                Defaults to (1,1).
            *args: Parameters for the CUDA function.

        Returns:
            list: List with each one of the outputs from the CUDA function

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

        processed_args: list[np.generic | np.ndarray] = []
        gpu_args: list[np.generic | np.ndarray] = []
        output_results: list = []
        # CUDA source code collecting all device and global functions 
        includes: set[str] = {
            include for program in self.src_code.values() 
            for include in program.includes
        }
        funcs: list[str] = [
            program.functions for program in self.src_code.values()
        ]
        source = "\n".join(list(includes) + funcs)
        
        module = SourceModule(source)
        
        kernel = module.get_function(func_name)
        for i, argument in enumerate(args):
            # Prepare Kernel Outputs
            if i in outputs_idx: 
                shape, dtype = outputs_details[i]
                out = gpuarray.empty(shape, dtype)
                processed_args.append(out)
                output_results.append(out)
            # Prepare Input Parameters
            else:
                self._process_argument(argument, processed_args)

        gpu_args = [
            arg.gpudata if isinstance(arg, gpuarray.GPUArray) 
            else arg for arg in processed_args
        ]

        # PyCUDA execution
        kernel(*gpu_args, block=block, grid=grid)
        
 
        return [output.get() for output in output_results]
        
    def single_operation(self, func_name: str, *args) -> Any:
        """
        Performs a simple operation using CUDA.
        
        Provides access to GPU-accelerated functions defined by 'pycuda.cumath'.
        Handles any GPU-compatible processing needed for its execution. 

        Args:
            func_name (str): Name of the desired CUDA operation implemented 
                by 'pycuda.cumath'.
            *args: List of arguments needed for the desired operation that will
                be processed for GPU-compatibility.

        Returns:
            Any: A host copy of the result of the CUDA operation

        Raises:
            AttributeError: If 'func_name' does not exist in 'pycuda.cumath'
        """

        if hasattr(pycuda.cumath, func_name):
            func = getattr(pycuda.cumath, func_name)
            gpu_args: list = []
            for arg in args:
                self._process_argument(arg, gpu_args)
            return func(*gpu_args).get()
        else:
            raise AttributeError(
                f"Operation '{func_name}' not implemented by pycuda.cumath."
            )

    def reduction_operation(self, op_name: str, array: Any) -> Any:
        """
        Performs a reduction operation for an array using 'pycuda.gpuarray'.

        Provides access to reduction operations (sum, max, min, etc.) defined 
        by 'pycuda.gpuarray'. Handles any GPU-compatible processing needed for 
        its execution. 

        Args:
            op_name (str): Name of the desired reduction operation implemented 
                by 'pycuda.gpuarray'.
            array (Any): Data array to be reduced.

        Returns:
            Any: A host copy of the result of the reduction operation.

        Raises:
            AttributeError: If 'op_name' does not exist in 'pycuda.gpuarray'
        """
        if not isinstance(array, gpuarray.GPUArray):
            array = gpuarray.to_gpu(array)

        if hasattr(pycuda.gpuarray, op_name):
            reduct = getattr(pycuda.gpuarray, op_name)
            try:
                result = reduct(array)
            except TypeError:
                result = reduct()(array)
            return result.get()
        else:
            raise AttributeError(
                f"Operation '{op_name}' not implemented by pycuda.gpuarray."
            )

    @singledispatchmethod
    def _process_argument(self, arg, gpu_args: list) -> None:
        """
        Prepares parameters for GPU execution.
        
        Args:
            arg (Any): argument to be processed.
            gpu_args (list): list where the processed argument will be added if
                admitted.
        Raises:
            TypeError: If the provided argument type is not supported.
        """
        raise TypeError(
            f"Type {type(arg)} not admitted for a function"
        )

    @_process_argument.register(np.ndarray)
    def _(self, arg: np.ndarray, gpu_args: list) -> None:
        gpu_args.append(gpuarray.to_gpu(arg))

    @_process_argument.register(np.integer)
    def _(self, arg: np.integer, gpu_args: list) -> None:
        if (isinstance(arg, (np.int32, np.int64))):
            gpu_args.append(arg)
        else:
            raise TypeError(
                f"Type {type(arg)} not admitted for a function"
            )

    @_process_argument.register(np.floating)
    def _(self, arg: np.floating, gpu_args: list) -> None:
        if (isinstance(arg, (np.float32, np.float64))):
            gpu_args.append(arg)
        else:
            raise TypeError(
                f"Type {type(arg)} not admitted for a function"
            )
        
    @_process_argument.register(int)
    def _(self, arg: int, gpu_args: list) -> None:
        gpu_args.append(np.int32(arg))

    @_process_argument.register(float)
    def _(self, arg: float, gpu_args: list) -> None:
        gpu_args.append(np.float64(arg))

    @_process_argument.register(list)
    def _(self, arg: list, gpu_args: list) -> None:
        array = np.array(arg)
        gpu_args.append(gpuarray.to_gpu(array))

