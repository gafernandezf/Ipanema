
import atexit
import pycuda.autoinit

from sdk.cuda_manager.implementations.pycuda_cuda_manager import PyCudaManager


class AutoCudaManager(PyCudaManager):
    
    def __init__(self):
        super().__init__()
        
    def _initialize_context(self):
        pass
    
    def _finish_up_context(self):
        pass