import tinytorch
from hypothesis import given
from .strategies import tensors, assert_close
import pytest


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = tinytorch.avgpool2d(t, (2, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = tinytorch.avgpool2d(t, (2, 1))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = tinytorch.avgpool2d(t, (1, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    tinytorch.grad_check(lambda t: tinytorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t):
    assert tinytorch.max(t, 0) == max([t[0, 0, 0], t[1, 0, 0]])
    assert tinytorch.max(t, 1) == max([t[0, 0, 0], t[0, 1, 0], t[0, 2, 0]])
    assert tinytorch.max(t, 2) == max([t[0, 0, 0], t[0, 0, 1], t[0, 0, 2], t[0, 0, 3]])

    jitter = t + tinytorch.rand((2, 3, 4)) * 1e-2
    tinytorch.grad_check(lambda a: tinytorch.max(a, 0), jitter)
    tinytorch.grad_check(lambda a: tinytorch.max(a, 1), jitter)
    tinytorch.grad_check(lambda a: tinytorch.max(a, 2), jitter)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t):
    out = tinytorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = tinytorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = tinytorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t):
    q = tinytorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = tinytorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = tinytorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    q = tinytorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = tinytorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    tinytorch.grad_check(lambda a: tinytorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t):
    q = tinytorch.softmax(t, 3)
    q2 = tinytorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    tinytorch.grad_check(lambda a: tinytorch.logsoftmax(a, dim=2), t)