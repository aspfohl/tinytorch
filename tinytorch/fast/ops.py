import numpy as np
from numba import njit, prange

from ..tensor.data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
j_to_index = njit(inline="always")(to_index)
j_index_to_position = njit(inline="always")(index_to_position)
j_broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        if np.array_equal(in_strides, out_strides) and np.array_equal(
            in_shape, out_shape
        ):
            # if strides are identical, skip using indexing entirely
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            for idx in prange(len(out)):
                in_index = np.empty(MAX_DIMS, np.int32)
                out_index = np.empty(MAX_DIMS, np.int32)
                j_to_index(idx, out_shape, out_index)
                j_broadcast_index(out_index, out_shape, in_shape, in_index)
                in_pos = j_index_to_position(in_index, in_strides)
                out_pos = j_index_to_position(out_index, out_strides)
                out[out_pos] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        if (
            np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(a_shape, b_shape)
        ):
            # if strides are identical, skip using indexing entirely
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for idx in prange(len(out)):
                out_index = np.empty(MAX_DIMS, np.int32)
                a_index = np.empty(MAX_DIMS, np.int32)
                b_index = np.empty(MAX_DIMS, np.int32)

                j_to_index(idx, out_shape, out_index)
                out_pos = j_index_to_position(out_index, out_strides)

                j_broadcast_index(out_index, out_shape, a_shape, a_index)
                a_pos = j_index_to_position(a_index, a_strides)

                j_broadcast_index(out_index, out_shape, b_shape, b_index)
                b_pos = j_index_to_position(b_index, b_strides)

                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function.

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`

    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        reduce_size = a_shape[reduce_dim]
        reduce_stride = a_strides[reduce_dim]
        for idx in prange(len(out)):
            out_index = np.empty(MAX_DIMS, np.int32)
            # to_index(idx, out_shape, out_index)
            cur_ord = idx + 0
            for i in range(len(out_shape) - 1, -1, -1):
                s = out_shape[i]
                out_index[i] = int(cur_ord % s)
                cur_ord = cur_ord // s
            out_pos = j_index_to_position(out_index, out_strides)

            a_pos = 0
            for j in range(len(a_shape)):
                if j != reduce_dim:
                    a_pos += out_index[j] * a_strides[j]

            temp = out[out_pos]
            for a in range(a_pos, reduce_stride * reduce_size + a_pos, reduce_stride):
                temp = fn(temp, a_storage[a])
            out[out_pos] = temp

    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


@njit(parallel=True, fastmath=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Optimizations:

        * Outer loop in parallel
        * No index buffers or function calls
        * Inner loop should have no global writes, 1 multiply.


    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_s = a_strides[0] if a_shape[0] > 1 else 0
    b_s = b_strides[0] if b_shape[0] > 1 else 0
    for i in prange(len(out)):
        # produce out_pos, and b (batch), similar to index_to_position and to_index combined
        temp, out_pos = i + 0, 0
        for j in range(len(out_shape) - 1, -1, -1):
            b = int(temp % out_shape[j])
            out_pos += b * out_strides[j]
            temp //= out_shape[j]

        # calculate the position for a and b
        temp = out_shape[len(out_shape) - 1]
        a_pos = a_s * b + int(i // temp % out_shape[len(out_shape) - 2]) * a_strides[-2]
        b_pos = b_s * b + int(i % temp) * b_strides[-1]

        # iterate over the last dim, multiplying a by b and incrementing position for a and b
        temp = 0
        for _ in range(a_shape[-1]):
            temp += a_storage[a_pos] * b_storage[b_pos]
            a_pos += a_strides[-1]
            b_pos += b_strides[-2]
        out[out_pos] = temp


def matrix_multiply(a, b):
    """
    Batched tensor matrix multiply ::

        for n:
          for i:
            for j:
              for k:
                out[n, i, j] += a[n, i, k] * b[n, k, j]

    Where n indicates an optional broadcasted batched dimension.

    Should work for tensor shapes of 3 dims ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b

    Returns:
        :class:`Tensor` : new tensor data
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
