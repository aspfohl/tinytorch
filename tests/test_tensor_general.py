import tinytorch
import random
from tinytorch import grad_check
import pytest
from hypothesis import given, settings
import numba
from hypothesis.strategies import integers, lists, data, permutations
from .strategies import (
    tensors,
    shaped_tensors,
    assert_close,
    assert_close_tensor,
    small_floats,
)
from tinytorch import MathTestVariable

one_arg, two_arg, red_arg = MathTestVariable._tests()


# The tests in this file only run the main mathematical functions.
# The difference is that they run with different tensor ops backends.

TensorBackend = tinytorch.make_tensor_backend(tinytorch.TensorOps)
FastTensorBackend = tinytorch.make_tensor_backend(tinytorch.FastOps)
shared = {"fast": FastTensorBackend}

# todo: combine
backend_tests = [pytest.param("fast")]
matmul_tests = [pytest.param("fast")]


if numba.cuda.is_available():
    backend_tests.append(pytest.param("cuda"))

    matmul_tests.append(pytest.param("cuda"))
    shared["cuda"] = tinytorch.make_tensor_backend(tinytorch.CudaOps, is_cuda=True)


@given(lists(small_floats, min_size=1))
@pytest.mark.parametrize("backend", backend_tests)
def test_create(backend, t1):
    "Create different tensors."
    t2 = tinytorch.tensor(t1, backend=shared[backend])
    for i in range(len(t1)):
        assert t1[i] == t2[i]


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_one_args(fn, backend, data):
    "Run forward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t2 = tensor_fn(t1)
    for ind in t2._tensor.indices():
        assert_close(t2[ind], base_fn(t1[ind]))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_args(fn, backend, data):
    "Run forward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t3 = tensor_fn(t1, t2)
    for ind in t3._tensor.indices():
        assert_close(t3[ind], base_fn(t1[ind], t2[ind]))


@given(data())
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_one_derivative(fn, backend, data):
    "Run backward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data())
@settings(max_examples=50)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_grad(fn, backend, data):
    "Run backward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1, t2)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", red_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_reduce(fn, backend, data):
    "Run backward for all reduce functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


if numba.cuda.is_available():

    def test_sum_practice():
        x = [random.random() for i in range(16)]
        b = tinytorch.tensor(x)
        s = b.sum()[0]
        b2 = tinytorch.tensor(x, backend=shared["cuda"])
        out = tinytorch.sum_practice(b2)
        assert_close(s, out._storage[0])

    def test_sum_practice2():
        x = [random.random() for i in range(64)]
        b = tinytorch.tensor(x)
        s = b.sum()[0]
        b2 = tinytorch.tensor(x, backend=shared["cuda"])
        out = tinytorch.sum_practice(b2)
        assert_close(s, out._storage[0] + out._storage[1])

    def test_sum_practice3():
        x = [random.random() for i in range(48)]
        b = tinytorch.tensor(x)
        s = b.sum()[0]
        b2 = tinytorch.tensor(x, backend=shared["cuda"])
        out = tinytorch.sum_practice(b2)
        assert_close(s, out._storage[0] + out._storage[1])

    def test_sum_practice4():
        x = [random.random() for i in range(32)]
        b = tinytorch.tensor(x)
        s = b.sum()[0]
        b2 = tinytorch.tensor(x, backend=shared["cuda"])
        out = b2.sum(0)
        assert_close(s, out[0])

    def test_sum_practice5():
        x = [random.random() for i in range(500)]
        b = tinytorch.tensor(x)
        s = b.sum()[0]
        b2 = tinytorch.tensor(x, backend=shared["cuda"])
        out = b2.sum(0)
        assert_close(s, out[0])

    def test_sum_practice_other_dims():
        x = [[3 * j + i + 1 for i in range(4)] for j in range(3)]
        b = tinytorch.tensor(x)
        s = b.sum(1)
        b2 = tinytorch.tensor(x, backend=shared["cuda"])
        out = b2.sum(1)
        print(out)
        for i in range(16):
            print(i, s[i, 0], out[i, 0])
            assert_close(s[i, 0], out[i, 0])

    def test_sum_practice_other_dims2():
        x = [[3 * j + i + 1 for i in range(4)] for j in range(3)]

        b = tinytorch.tensor(x)
        print(b)
        s = b.sum(0)
        b2 = tinytorch.tensor(x, backend=shared["cuda"])
        out = b2.sum(0)
        print(out)
        for i in range(4):
            print(i, s[0, i], out[0, i])
            assert_close(s[0, i], out[0, i])

    def test_mul_practice1():
        x = [[random.random() for i in range(2)] for j in range(2)]
        y = [[random.random() for i in range(2)] for j in range(2)]
        z = tinytorch.tensor(x, backend=shared["fast"]) @ tinytorch.tensor(
            y, backend=shared["fast"]
        )

        x = tinytorch.tensor(x, backend=shared["cuda"])
        y = tinytorch.tensor(y, backend=shared["cuda"])
        z2 = tinytorch.mm_practice(x, y)
        for i in range(2):
            for j in range(2):
                assert_close(z[i, j], z2._storage[2 * i + j])

    def test_mul_practice2():
        x = [[random.random() for i in range(32)] for j in range(32)]
        y = [[random.random() for i in range(32)] for j in range(32)]
        z = tinytorch.tensor(x, backend=shared["fast"]) @ tinytorch.tensor(
            y, backend=shared["fast"]
        )

        x = tinytorch.tensor(x, backend=shared["cuda"])
        y = tinytorch.tensor(y, backend=shared["cuda"])
        z2 = tinytorch.mm_practice(x, y)
        for i in range(32):
            for j in range(32):
                assert_close(z[i, j], z2._storage[32 * i + j])

    def test_mul_practice3():
        "Small real example"
        x = [[random.random() for i in range(2)] for j in range(2)]
        y = [[random.random() for i in range(2)] for j in range(2)]
        z = tinytorch.tensor(x, backend=shared["fast"]) @ tinytorch.tensor(
            y, backend=shared["fast"]
        )

        x = tinytorch.tensor(x, backend=shared["cuda"])
        y = tinytorch.tensor(y, backend=shared["cuda"])
        z2 = x @ y

        for i in range(2):
            for j in range(2):
                assert_close(z[i, j], z2[i, j])

    def test_mul_practice4():
        "Extend to require 2 blocks"
        size = 33
        x = [[random.random() for i in range(size)] for j in range(size)]
        y = [[random.random() for i in range(size)] for j in range(size)]
        z = tinytorch.tensor(x, backend=shared["fast"]) @ tinytorch.tensor(
            y, backend=shared["fast"]
        )

        x = tinytorch.tensor(x, backend=shared["cuda"])
        y = tinytorch.tensor(y, backend=shared["cuda"])
        z2 = x @ y

        for i in range(size):
            for j in range(size):
                assert_close(z[i, j], z2[i, j])

    def test_mul_practice5():
        "Extend to require a batch"
        size = 33
        x = [
            [[random.random() for i in range(size)] for j in range(size)]
            for _ in range(2)
        ]
        y = [
            [[random.random() for i in range(size)] for j in range(size)]
            for _ in range(2)
        ]
        z = tinytorch.tensor(x, backend=shared["fast"]) @ tinytorch.tensor(
            y, backend=shared["fast"]
        )

        x = tinytorch.tensor(x, backend=shared["cuda"])
        y = tinytorch.tensor(y, backend=shared["cuda"])
        z2 = x @ y

        for b in range(2):
            for i in range(size):
                for j in range(size):
                    assert_close(z[b, i, j], z2[b, i, j])

    def test_mul_practice6():
        "Extend to require a batch"
        size_a = 45
        size_b = 40
        size_in = 33
        x = [
            [[random.random() for i in range(size_in)] for j in range(size_a)]
            for _ in range(2)
        ]
        y = [
            [[random.random() for i in range(size_b)] for j in range(size_in)]
            for _ in range(2)
        ]
        z = tinytorch.tensor(x, backend=shared["fast"]) @ tinytorch.tensor(
            y, backend=shared["fast"]
        )

        x = tinytorch.tensor(x, backend=shared["cuda"])
        y = tinytorch.tensor(y, backend=shared["cuda"])
        z2 = x @ y

        for b in range(2):
            for i in range(size_a):
                for j in range(size_b):
                    print(i, j)
                    assert_close(z[b, i, j], z2[b, i, j])


@given(data())
@settings(max_examples=25)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_grad_broadcast(fn, backend, data):
    "Run backward for all two arg functions above with broadcast."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn

    grad_check(tensor_fn, t1, t2)

    # broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("backend", backend_tests)
def test_permute(backend, data):
    "Check permutations for all backends."
    t1 = data.draw(tensors(backend=shared[backend]))
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a):
        return a.permute(*permutation)

    tinytorch.grad_check(permute, t1)


def test_mm2():
    a = tinytorch.rand((2, 3), backend=FastTensorBackend)
    b = tinytorch.rand((3, 4), backend=FastTensorBackend)
    c = a @ b

    c2 = (a.view(2, 3, 1) * b.view(1, 3, 4)).sum(1).view(2, 4)

    for ind in c._tensor.indices():
        assert_close(c[ind], c2[ind])

    tinytorch.grad_check(lambda a, b: a @ b, a, b)


# Matrix Multiplication


@given(data())
@pytest.mark.parametrize("backend", matmul_tests)
def test_bmm(backend, data):
    small_ints = integers(min_value=2, max_value=4)
    A, B, C, D = (
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
    )
    a = data.draw(tensors(backend=shared[backend], shape=(D, A, B)))
    b = data.draw(tensors(backend=shared[backend], shape=(1, B, C)))

    c = a @ b
    c2 = (
        (a.contiguous().view(D, A, B, 1) * b.contiguous().view(1, 1, B, C))
        .sum(2)
        .view(D, A, C)
    )
    assert_close_tensor(c, c2)
