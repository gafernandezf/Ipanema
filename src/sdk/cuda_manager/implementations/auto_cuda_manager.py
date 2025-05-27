# The import of 'pycuda.autoinit' itself handles the context
import pycuda.autoinit
from sdk.cuda_manager.implementations.pycuda_cuda_manager import PyCudaManager


class AutoCudaManager(PyCudaManager):
    """Cuda Handler which uses PyCuda's Automatic Context."""
        
    def _initialize_context(self):
        """
        Initializes CUDA context.
        
        Since this Cuda Manager is using 'pycuda.autoinit' extra initialization
        logic is not needed.
        """
        pass
    
    def _finish_up_context(self):
        """
        Cleans the CUDA context.
        
        Since this Cuda Manager is using 'pycuda.autoinit' extra deletion
        logic is not needed.
        """
        pass