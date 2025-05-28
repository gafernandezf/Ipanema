from pathlib import Path
from sdk.cuda_manager.abstract_cuda_manager import CudaManager
from sdk.cuda_manager.implementations.auto_cuda_manager import AutoCudaManager
from sdk.math_utils.genetic.abstract_genetic_algorithm import (
    AbstractGeneticAlgorithm
)

class GeneticAlgorithm(AbstractGeneticAlgorithm):

    cuda_manager: CudaManager

    def __init__(self):
        super().__init__()
        self.cuda_manager = AutoCudaManager()

        self.cuda_manager.add_code_fragment(
            "genetic",
            Path(
                r"src\sdk\math_utils\genetic\_support_files\_impl_genetic.cu"
            )
        )

    def real_selection():
        pass

    def complex_selection():
        pass

    def real_mutation():
        pass

    def complex_mutation():
        pass