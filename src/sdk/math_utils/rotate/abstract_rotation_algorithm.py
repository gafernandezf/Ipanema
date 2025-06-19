from abc import ABC, abstractmethod
import numpy as np

class AbstractRotationAlgorithm(ABC):
    """
    Abstract interface for rotation or linear transformation strategies.

    This class defines the required method for implementing transformations 
    of float32 vectors or matrices using a transformation matrix 'T'.
    """
    @abstractmethod
    def transform_f32(
        self,
        in_matrix: np.ndarray, 
        t_matrix: np.ndarray,
        n: int
    ) -> np.array[np.double]:
        """
        Applies a float32 transformation using the selected strategy.

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
        pass
