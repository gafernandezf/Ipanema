import numpy as np
import pytest
from unittest import mock

from sdk.cuda_manager.implementations.pycuda_cuda_manager import PyCudaManager

class FakeCudaManager(PyCudaManager):
    def _initialize_context(self):
        pass

    def _finish_up_context(self):
        pass


####################
# _process_argument
####################


def test_process_argument_ndarray():
    manager = FakeCudaManager()
    gpu_args = []
    array = np.array([1, 2, 3], dtype=np.float32)

    with mock.patch("pycuda.gpuarray.to_gpu") as mocked_to_gpu:
        manager._process_argument(array, gpu_args)
        mocked_to_gpu.assert_called_once_with(array)

@pytest.mark.parametrize("arg,expected_type", [
    (5, np.int32),
    (3.14, np.float64),
    (np.int32(42), np.int32),
    (np.float64(1.23), np.float64)
])

def test_process_argument_scalars(arg, expected_type):
    manager = FakeCudaManager()
    gpu_args = []
    manager._process_argument(arg, gpu_args)
    assert isinstance(gpu_args[0], expected_type)


###################
# single_operation
###################


def test_single_operation_calls_cumath():
    manager = FakeCudaManager()
    dummy_gpuarray = mock.Mock()
    dummy_gpuarray.get.return_value = "result"

    with mock.patch("pycuda.cumath.sin", return_value=dummy_gpuarray) as mocked_func:
        with mock.patch.object(manager, "_process_argument") as mocked_proc:
            mocked_proc.side_effect = lambda arg, gpu_args: gpu_args.append(arg)
            result = manager.single_operation("sin", np.array([1, 2, 3]))

    mocked_func.assert_called_once()
    assert result == "result"


######################
# reduction_operation
######################


@mock.patch("pycuda.gpuarray.sum")
@mock.patch("pycuda.gpuarray.to_gpu")
def test_reduction_operation_mocks_gpu(to_gpu_mock, sum_mock):
    manager = FakeCudaManager()

    gpu_array_mock = mock.Mock()
    to_gpu_mock.return_value = gpu_array_mock

    result_mock = mock.Mock()
    result_mock.get.return_value = 42.0
    sum_mock.return_value = result_mock

    result = manager.reduction_operation("sum", [1.0, 2.0, 3.0])

    assert result == 42.0
    to_gpu_mock.assert_called_once()
    sum_mock.assert_called_once_with(gpu_array_mock)
    result_mock.get.assert_called_once()