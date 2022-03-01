import pytest
from hypothesis import given

from tests.strategies import assert_close, small_floats, small_scalars
from tinytorch import operators
from tinytorch.scalar import Scalar, central_difference, derivative_check
from tinytorch.testing import MathTestVariable


def test_central_diff():
    d = central_difference(operators.id, 5, arg=0)
    assert_close(d, 1.0)

    d = central_difference(operators.add, 5, 10, arg=0)
    assert_close(d, 1.0)

    d = central_difference(operators.mul, 5, 10, arg=0)
    assert_close(d, 10.0)

    d = central_difference(operators.mul, 5, 10, arg=1)
    assert_close(d, 5.0)

    d = central_difference(operators.exp, 2, arg=0)
    assert_close(d, operators.exp(2.0))


@given(small_floats, small_floats)
def test_simple(a, b):
    # Simple add
    c = Scalar(a) + Scalar(b)
    assert_close(c.data, a + b)

    # Simple mul
    c = Scalar(a) * Scalar(b)
    assert_close(c.data, a * b)

    # Simple relu
    c = Scalar(a).relu() + Scalar(b).relu()
    assert_close(c.data, operators.relu(a) + operators.relu(b))

    # Add others if you would like...


one_arg, two_arg, _ = MathTestVariable._tests()


@given(small_scalars)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    name, base_fn, scalar_fn = fn
    assert_close(scalar_fn(t1).data, base_fn(t1.data))


@given(small_scalars, small_scalars)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, t1, t2):
    name, base_fn, scalar_fn = fn
    assert_close(scalar_fn(t1, t2).data, base_fn(t1.data, t2.data))


# See tinytorch.testing for all of the functions checked.


@given(small_scalars)
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(fn, t1):
    name, _, scalar_fn = fn
    derivative_check(scalar_fn, t1)


@given(small_scalars, small_scalars)
@pytest.mark.parametrize("fn", two_arg)
def test_two_derivative(fn, t1, t2):
    name, _, scalar_fn = fn
    derivative_check(scalar_fn, t1, t2)


def test_scalar_name():
    x = Scalar(10, name="x")
    y = (x + 10.0) * 20
    y.name = "y"
    return y
