import pytest
from hypothesis import given
from hypothesis.strategies import lists
from minitorch import MathTest
from minitorch.operators import (
    add,
    addLists,
    eq,
    exp,
    id,
    inv,
    inv_back,
    is_close,
    log,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
    zipWith,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x, y):
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if x != 0.0:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@pytest.mark.task0_2
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert relu(a) == a
    if a <= 0:
        assert relu(a) == 0.0

    assert relu(a) in {0, a}
    assert 0.0 <= relu(a)


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a, b):
    if a > 0:
        assert relu_back(a, b) == b
    if a <= 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a):
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a):
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a):
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a):
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a):
    siga = sigmoid(a)

    # It is always between 0.0 and 1.0
    assert 0.0 <= siga <= 1.0

    # one minus sigmoid is the same as negative sigmoid
    assert_close(1 - siga, sigmoid(-a))

    # it is  strictly increasing
    left = sigmoid(a - 1)
    right = sigmoid(a + 1)
    if not is_close(left, right):  # avoids cases of python float drift
        assert left < siga < right
    assert left <= siga <= right  # regardless of python floats, this will hold


@pytest.mark.task0_2
def test_sigmoid_zero():
    """
    Special property of sigmoid at zero
    """
    assert sigmoid(0) == 0.5


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    # if a < b and b < c then a < c
    if lt(a, b) and lt(b, c):
        assert lt(a, c) == 1.0


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a, b):
    assert mul(a, b) == mul(b, a)
    assert max(a, b) == max(b, a)
    assert add(a, b) == add(b, a)
    assert eq(a, b) == eq(b, a)
    assert is_close(a, b) == is_close(b, a)


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_not_symmetric(a, b):
    if eq(a, b):
        # actually symmetric when a==b
        assert lt(a, b) == 0.0
        assert lt(b, a) == 0.0
    else:
        assert lt(a, b) != lt(b, a)
        assert lt(a, b) + lt(b, a) == 1.0


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(a, b, c):
    assert_close(mul(a, add(b, c)), add(mul(a, b), mul(a, c)))


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_is_close(a, b):
    if eq(a, b):
        assert is_close(a, b)


@pytest.mark.task0_2
@given(small_floats)
def test_neg(a):
    assert neg(a) == mul(-1, a)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_max_distributive(a, b, c):
    assert max(max(a, b), c) == max(max(b, c), a) == max(max(a, c), b)


@pytest.mark.task0_2
@given(small_floats)
def test_log(a):
    if a < 0 and not is_close(a, 0):
        with pytest.raises(ValueError):
            log(a)
    elif 0 <= a <= 1:
        assert log(a) < 0
    elif a == 1:
        assert_close(log(a), 0)
    else:
        assert log(a) > 0


@pytest.mark.task0_2
@given(small_floats)
def test_exp(a):
    assert 0 < exp(a)


@pytest.mark.task0_2
def test_exp_zero():
    assert exp(0) == 1.0


@pytest.mark.task0_2
@given(small_floats)
def test_log_back(a):
    if a == 0:
        with pytest.raises(ZeroDivisionError):
            log_back(a, 1)
    else:
        assert log_back(a, 1) == inv(a)


@pytest.mark.task0_2
@given(small_floats)
def test_inv(a):
    if a == 0:
        with pytest.raises(ZeroDivisionError):
            inv(a)
    else:
        assert_close(inv(a) * a, 1.0)


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_inv_back(a, b):
    if a == 0:
        with pytest.raises(ZeroDivisionError):
            inv_back(a, b)
    elif b < 0:
        assert inv_back(a, b) > 0
    elif b == 0:
        assert inv_back(a, b) == 0
    else:
        assert inv_back(a, b) < 0


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1, ls2):
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    assert_close(sum(addLists(ls1, ls2)), add(sum(ls1), sum(ls2)))


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert prod([1, 2, 3]) == 6
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_map(ls):
    pass


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls):
    check = negList(ls)
    for i in range(len(ls)):
        assert_close(check[i], -ls[i])


@pytest.mark.task0_3
def test_zipwith():
    mulLists = zipWith(mul)
    assert mulLists([], []) == []
    assert mulLists([5], [3, 2]) == [15]


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    name, base_fn, _ = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, t1, t2):
    name, base_fn, _ = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a, b):
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
