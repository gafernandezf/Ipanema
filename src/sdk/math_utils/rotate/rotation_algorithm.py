from pathlib import Path
import numpy as np
from sdk.cuda_manager.abstract_cuda_manager import CudaManager
from sdk.cuda_manager.implementations.auto_cuda_manager import AutoCudaManager
from sdk.math_utils.rotate.abstract_rotation_algorithm import (
    AbstractRotationAlgorithm
)

class RotationAlgorithm(AbstractRotationAlgorithm):
    """
    CUDA-based implementation of a float32 matrix rotation algorithm.

    This class implements the AbstractRotationAlgorithm interface, providing
    a GPU-accelerated matrix transformation using a CUDA kernel. It delegates
    execution to a CudaManager and expects CUDA-compatible inputs.

    The transformation applies a linear operation to each row of the input
    matrix using a transformation matrix, effectively performing a rotation
    or projection in float32 precision.
    """

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

    def transform_f32(
        self,
        in_matrix: np.ndarray, 
        t_matrix: np.ndarray,
        n: int
    ) -> np.array[np.double]:
        """
        Applies the rotation transformation on the GPU using float32 inputs.

        Args:
            in_matrix (np.ndarray): Input matrix of shape (M, N) containing 
                float32 values.
            t_matrix (np.ndarray): Transformation matrix of shape (N, N) 
                containing float32 values.
            n (int): Dimension size used in the transformation computation.

        Returns:
            np.ndarray: Output matrix of shape (M, N) resulting from applying 
                the transformation.
        """
        transform_f32_out: list = self.cuda_manager.run_program(
            "transform_f32",
            [1],
            {1: [(len(in_matrix),), np.double]},
            (1, 1, 1),
            (int(n), 1, 1),
            in_matrix, 
            np.empty_like(in_matrix),
            t_matrix, 
            n
        )
        return transform_f32_out[0]
