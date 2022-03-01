from hypothesis import given

from tests.strategies import assert_close, tensors
from tinytorch import nn
from tinytorch.tensor.functions import grad_check


@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = nn.avgpool2d(t, (2, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = nn.avgpool2d(t, (2, 1))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = nn.avgpool2d(t, (1, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    grad_check(lambda t: nn.avgpool2d(t, (2, 2)), t)


@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t):
    out = nn.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = nn.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = nn.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@given(tensors())
def test_drop(t):
    q = nn.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = nn.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = nn.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    q = nn.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = nn.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    grad_check(lambda a: nn.softmax(a, dim=2), t)


@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t):
    q = nn.softmax(t, 3)
    q2 = nn.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    grad_check(lambda a: nn.logsoftmax(a, dim=2), t)
