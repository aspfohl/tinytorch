import tinytorch
import pytest
import numba
from hypothesis import given, settings
from .strategies import tensors

BACKEND = {"fast1": tinytorch.Conv1dFun, "fast2": tinytorch.Conv2dFun}
backend_1d = [pytest.param("fast")]
backend_2d = [pytest.param("fast")]

if numba.cuda.is_available():
    backend_1d.append(pytest.param("cuda"))
    backend_2d.append(pytest.param("cuda"))

    BACKEND["cuda1"] = tinytorch.CudaConv1dFun
    BACKEND["cuda2"] = tinytorch.CudaConv2dFun


@pytest.mark.parametrize("backend", backend_1d)
def test_conv1d_simple(backend):
    t = tinytorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = tinytorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    out = BACKEND[f"{backend}1"].apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
@pytest.mark.parametrize("backend", backend_1d)
def test_conv1d(backend, input, weight):
    tinytorch.grad_check(BACKEND[f"{backend}1"].apply, input, weight)


@settings(max_examples=50)
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@pytest.mark.parametrize("backend", backend_1d)
def test_conv1d_channel(backend, input, weight):
    tinytorch.grad_check(BACKEND[f"{backend}1"].apply, input, weight)


@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@pytest.mark.parametrize("backend", backend_2d)
def test_conv(backend, input, weight):
    tinytorch.grad_check(BACKEND[f"{backend}2"].apply, input, weight)


@settings(max_examples=10)
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@pytest.mark.parametrize("backend", backend_2d)
def test_conv_batch(backend, input, weight):
    tinytorch.grad_check(BACKEND[f"{backend}2"].apply, input, weight)


@settings(max_examples=10)
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@pytest.mark.parametrize("backend", backend_2d)
def test_conv_channel(backend, input, weight):
    tinytorch.grad_check(BACKEND[f"{backend}2"].apply, input, weight)


@pytest.mark.parametrize("backend", backend_2d)
def test_conv2(backend):
    t = tinytorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t.requires_grad_(True)

    t2 = tinytorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = BACKEND[f"{backend}2"].apply(t, t2)
    out.sum().backward()

    tinytorch.grad_check(BACKEND[f"{backend}2"].apply, t, t2)
