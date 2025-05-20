from pathlib import Path
from typing import Any, Tuple
from functools import singledispatchmethod
from pycuda.compiler import SourceModule
import pycuda.cumath
from sdk.cuda_manager import CudaManager
# El propio import de autoinit crea un contexto automatico
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

class AutoCudaManager(CudaManager):
    """Wrapper class for PyCuda's Automatic Context"""

    def run_program(self,
            func_name: str,
            outputs_idx: list[int],
            outputs_details: dict[
                int, Tuple[Tuple[int, ...], Any]
            ],
            block: Tuple[int, int, int] = (256,1,1),
            grid: Tuple[int, int] = (1,1),
            *args
    ) -> list:
        """Runs a specific sequence of CUDA functions."""

        # TODO mediciones de tiempo (en core?)

        processed_args = list[np.generic | np.ndarray] = []
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
        
        Gives access to functions defined by 'pycuda.cumath'.

        Args:
            func_name (str): name of the desired operation
            *args: arguments needed for the desired operation
        """

        # TODO procesar argumentos para gpu

        if hasattr(pycuda.cumath, func_name):
            func = getattr(pycuda.cumath, func_name)
            gpu_args: list = [
                self._process_argument(arg, gpu_args) for arg in args
            ]
            return func(gpu_args)
        else:
            raise AttributeError(
                f"Operation '{func_name}' not implemented by pycuda.cumath."
            )
        
    def reduction_operation(self, op_name: str, array: Any) -> Any:
        """Performs a reduction operation for a specific array"""
        # TODO implement
        pass

    @singledispatchmethod
    def _process_argument(self, arg, gpu_args: list) -> None:
        raise TypeError(
            f"Type {type(arg)} not admitted for a function"
        )

    @_process_argument.register(np.ndarray)
    def _(self, arg: np.ndarray, gpu_args: list) -> None:
        gpu_args.append(gpuarray.to_gpu(arg))

    @_process_argument.register(np.integer)
    def _(self, arg: np.integer, gpu_args: list) -> None:
        if (isinstance(np.int32, np.int64)):
            gpu_args.append(arg)
        else:
            raise TypeError(
                f"Type {type(arg)} not admitted for a function"
            )

    @_process_argument.register(np.floating)
    def _(self, arg: np.floating, gpu_args: list) -> None:
        if (isinstance(np.float32, np.float64)):
            gpu_args.append(arg)
        else:
            raise TypeError(
                f"Type {type(arg)} not admitted for a function"
            )
        
    @_process_argument.register(int)
    def _(self, arg: int, gpu_args: list) -> None:
        gpu_args.append(arg)

    @_process_argument.register(float)
    def _(self, arg: float, gpu_args: list) -> None:
        gpu_args.append(arg)