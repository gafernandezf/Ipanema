import atexit
from typing import Optional
from venv import logger
import pycuda.driver as cuda
from pycuda.tools import clear_context_caches
from reikna import cluda
from sdk.cuda_manager.implementations.pycuda_cuda_manager import PyCudaManager

class InteractiveCudaManager(PyCudaManager):
    """
    Cuda Handler with a PyCuda's Custom Context.

    Allows the user to select a specific device for GPU executions.
    """

    def __init__(self, idev: Optional[int]=None, interactive: bool=True):
        super().__init__()
        self._initialize_context(idev, interactive)
        atexit.register(self._finish_up_context)

    def _initialize_context(
            self, 
            idev: Optional[int]=None, 
            interactive: bool=True
        ):
        """
        Initialize CUDA. This function substitutes "make_default_context"
        function from PyCuda. It allows to correctly select the device to
        work. This function is meant to be called just once at the
        beginning of a script. Any other call will be ignored.

        Arguments:
            idev (int): device to work with. If "None", and 
                "interactive = False" use the first found.
            interactive (bool): determine whether to ask for a device or 
                to select the first device found

        Returns:
            None
        """
        
        cuda.init()

        ndev = cuda.Device.count()
        
        if ndev == 0:
            raise LookupError('No devices have been found')
        
        # Default device if anything fails
        defdev = 0
        
        if idev is not None:
            # Use the specified device
            if idev >= ndev:
                print(f'WARNING: Specified a device number ({idev}) greater \
                    than the maximum number of devices ({ndev}); \
                    set to {defdev}')
                idev = defdev
                
        elif not interactive:
            # Get the default device
            idev = defdev
            
        else:
            # Ask the user to select a device
            print(f'Found {ndev} available devices:')
            for i in range(ndev):
                print(f'- {cuda.Device(i).name()} [{i}]')
                
            idev = -1
            while True:
                try:
                    idev = input(f'Select a device (default {defdev}): ')
                    if idev.strip() == '':
                        idev = defdev
                    idev = int(idev)
                    if idev in range(ndev):
                        break
                except (ValueError, KeyboardInterrupt):
                    logger.warning(f"Invalid input {idev}")
        
        device = cuda.Device(int(idev))
        print(f'-- Using device "{device.name()}" [{idev}]')
        api = cluda.cuda_api()    
        
        self.device  = device
        self.context = device.make_context()
        self.thread  = api.Thread(self.context)

    def _finish_up_context(self):
        """
        Instructions to finalize the CUDA context. 
        
        This function mimics the context release behavior of 'pycuda.autoinit'.
        """
        if self.context is not None:
            logger.info(f"Finishing Up context '{self.context}'")
            try:
                self.context.pop()
                self.context.detach()
                clear_context_caches()
            except Exception as e:
                logger.error(f"Problem finishing up context: {e}")
            finally:
                self.context = None