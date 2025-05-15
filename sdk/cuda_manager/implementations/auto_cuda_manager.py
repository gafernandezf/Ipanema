from pathlib import Path
import numpy as np
from functools import singledispatchmethod
#import pycuda.autoinit
#import pycuda.driver as cuda
#from pycuda.compiler import SourceModule
#import pycuda.gpuarray as gpuarray
from sdk.cuda_manager.abstract_cuda_manager import CudaManager
from sdk.cuda_manager.cuda_function.cuda_function import CudaFunction

class AutoCudaManager(CudaManager):
    """Wrapper class for PyCuda's Automatic Mode"""

    source: str

    def run_program(self,
            func_name: str,
            args,
            outputs_idx: list[int],
            outputs_details: dict[int, dict[str, any]]) -> any:
        """Runs a specific sequence of CUDA functions."""

        # CUDA source code collecting all device and global functions 
        global_funcs: list[str] = [
            func.function for func in self.global_functions.values()
        ]
        device_funcs: list[str] = [
            func.function for func in self.device_functions.values()
        ]

        # TODO precompilacion para juntar todos los ficheros
        funcs: str = "\n".join(
            device_funcs +
            ["extern \"C\" {"] +
            global_funcs +
            ["}"]
        )

        self.source = funcs
        self.module = None # SourceModule(self.source)
        self.kernel = None # self.module.get_function(func_name)
        kernel_args = None

        cuda_args = []

        for i, argument in enumerate(args):
            # Prepare Kernel Outputs
            if i in outputs_idx: 

                pass
            # Prepare Input Parameters
            else:
                _process_argument(argument, cuda_args)

        # PyCUDA execution
        self.kernel(kernel_args, block=self.num_block)

        # Parsing Kernel Output
        kernel_results = []
        
        return kernel_results

        
@singledispatchmethod
def _process_argument(self, arg, cuda_args: list) -> None:
    raise TypeError(
        f"Type {type(arg)} not admitted for a function"
    )

@_process_argument.register(np.ndarray)
def _(self, arg: np.ndarray, cuda_args: list) -> None:
    cuda_args.append(None)

@_process_argument.register(np.integer)
def _(self, arg: np.integer, cuda_args: list) -> None:
    if (isinstance(np.int32, np.int64)):
        cuda_args.append(arg)
    else:
        raise TypeError(
            f"Type {type(arg)} not admitted for a function"
        )

@_process_argument.register(np.floating)
def _(self, arg: np.floating, cuda_args: list) -> None:
    if (isinstance(np.float32, np.float64)):
        cuda_args.append(arg)
    else:
        raise TypeError(
            f"Type {type(arg)} not admitted for a function"
        )