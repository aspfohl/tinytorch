import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    MAX_DIMS,
)
from .tensor_functions import Function
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
j_to_index = njit(inline="always")(to_index)
j_index_to_position = njit(inline="always")(index_to_position)
j_broadcast_index = njit(inline="always")(broadcast_index)


@njit(parallel=True)
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, _ = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    for o in prange(out_size):
        out_index = np.empty(MAX_DIMS, np.int32)
        j_to_index(o, out_shape, out_index)
        o0, o1, o2 = out_index[0], out_index[1], out_index[2]

        i_pos = j_index_to_position([o0, 0, o2], input_strides)
        w_pos = j_index_to_position([o1, 0, kw - 1 if reverse else 0], weight_strides)

        # iterate over in channels (from input)
        temp = 0
        for _ in range(in_channels):
            temp_i_pos = i_pos
            temp_w_pos = w_pos

            # iterate over slider size (from weight)
            for w in range(kw):
                if not 0 <= o2 + ((w - kw + 1) if reverse else w) < width:
                    continue

                temp += input[temp_i_pos] * weight[temp_w_pos]

                temp_i_pos += (1 if not reverse else -1) * input_strides[2]
                temp_w_pos += (1 if not reverse else -1) * weight_strides[2]

            i_pos += input_strides[1]
            w_pos += weight_strides[1]

        # save to output
        out[j_index_to_position(out_index, out_strides)] = temp


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


@njit(parallel=True, fastmath=True)
def tensor_conv2d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    for o in prange(out_size):
        out_index = np.empty(MAX_DIMS, np.int32)
        j_to_index(o, out_shape, out_index)
        o0, o1, o2, o3 = out_index[0], out_index[1], out_index[2], out_index[3]

        i_pos = j_index_to_position([o0, 0, o2, o3], input_strides)
        w_pos = j_index_to_position(
            [o1, 0, kh - 1 if reverse else 0, kw - 1 if reverse else 0], weight_strides,
        )

        # iterate over in channels (from input)
        multiplier = 1 if not reverse else -1
        temp = 0
        for _ in range(in_channels):
            temp_i_pos = i_pos
            temp_w_pos = w_pos

            # iterate over slider size (from weight)
            for h in range(kh):
                temp2_i_pos = temp_i_pos
                temp2_w_pos = temp_w_pos

                for w in range(kw):  # 2nd dimension
                    if (
                        not reverse and 0 <= o2 + h < height and 0 <= o3 + w < width
                    ) or (
                        reverse
                        and 0 <= o2 - kh + h + 1 < height
                        and 0 <= o3 - kw + w + 1 < width
                    ):
                        temp += input[temp2_i_pos] * weight[temp2_w_pos]
                        temp2_i_pos += multiplier * input_strides[3]
                        temp2_w_pos += multiplier * weight_strides[3]

                temp_i_pos += multiplier * input_strides[2]
                temp_w_pos += multiplier * weight_strides[2]

            i_pos += input_strides[1]
            w_pos += weight_strides[1]

        out[j_index_to_position(out_index, out_strides)] = temp


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
