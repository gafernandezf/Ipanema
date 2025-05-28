import pytest

from sdk.cuda_manager.abstract_cuda_manager import CudaManager

def test_CudaManager_cannot_be_instantiated():
    with pytest.raises(TypeError):
        CudaManager()