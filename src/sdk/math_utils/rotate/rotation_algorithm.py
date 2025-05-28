from pathlib import Path
from sdk.cuda_manager.abstract_cuda_manager import CudaManager
from sdk.cuda_manager.implementations.auto_cuda_manager import AutoCudaManager
from sdk.math_utils.rotate.abstract_rotation_algorithm import (
    AbstractRotationAlgorithm
)

class RotationAlgorithm(AbstractRotationAlgorithm):

    cuda_manager: CudaManager

    def __init__(self):
        super().__init__()
        self.cuda_manager = AutoCudaManager()

        self.cuda_manager.add_code_fragment(
            "rotate",
            Path(
                r"src\sdk\math_utils\rotate\_support_files\_impl_rotate.cu"
            )
        )

    def transform_f32():
        pass