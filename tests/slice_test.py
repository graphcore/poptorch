#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch

import poptorch
import helpers


def slice_test_harness(tensor_x, tensor_y, start_fn, end_fn, step):
    op = lambda x, y: x[start_fn(x):end_fn(x):step] + y

    model = helpers.ModelWithWeights(op, tensor_x.shape)

    # Run on CPU.
    native_out, _ = model((tensor_x, tensor_y))

    # Run on IPU.
    poptorch_model = poptorch.trainingModel(model)
    poptorch_out, _ = poptorch_model((tensor_x, tensor_y))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("step", [1, 2, 3])
def test_slice_idx_size_of(step):
    def start_fn(tensor_in):
        return tensor_in.shape[0] // 2

    def end_fn(tensor_in):
        return tensor_in.shape[0] - 1

    slice_test_harness(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                       torch.tensor([3.0]), start_fn, end_fn, step)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_slice_with_sum(step):
    def start_fn(tensor_in):
        del tensor_in
        return torch.sum(torch.tensor([1, 2, 3])) // 3 - 2

    def end_fn(tensor_in):
        del tensor_in
        return torch.sum(torch.tensor([1, 2, 3])) // 3 + 1

    slice_test_harness(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                       torch.tensor([-3.0]), start_fn, end_fn, step)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_slice_with_branch(step):
    def start_fn(tensor_in):
        del tensor_in
        a = torch.sum(torch.tensor([1, 2, 3])) // 3 - 2
        b = torch.sum(torch.tensor([3, 4, 5])) // 3 - 4
        return a + b + 1

    def end_fn(tensor_in):
        del tensor_in
        a = torch.sum(torch.tensor([3, 2, 1])) // 3 + 2
        b = torch.sum(torch.tensor([3, 4, 5])) // 3 + 1
        return a - 1 + b

    slice_test_harness(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                       torch.tensor([-3.0]), start_fn, end_fn, step)


def dynamic_slice_harness(tensor_in,
                          extra_in,
                          start_fn,
                          end_fn,
                          step,
                          test_training=True):

    if test_training:
        op = lambda t: t[start_fn(extra_in):end_fn(extra_in):step]
        model = helpers.ModelWithWeights(op, tensor_in.shape)

        # Run on CPU.
        native_out, _ = model((tensor_in, ))

        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model)
        poptorch_out, _ = poptorch_model((tensor_in, ))

        # Training test - check weights changed
        poptorch_model.assert_weights_changed()
    else:
        model = torch.nn.Module()
        model.forward = lambda t, e: t[start_fn(e):end_fn(e):step]

        # Run on CPU.
        native_out = model(tensor_in, extra_in)

        # Run on IPU.
        poptorch_model = poptorch.inferenceModel(model)
        poptorch_out = poptorch_model(tensor_in, extra_in)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_dynamic_slice_one_dim_add(step):
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 4

    dynamic_slice_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([1]), start_fn, end_fn, step)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_dynamic_slice_one_dim_subtract(step):
    def start_fn(extra_in):
        return extra_in - 4

    def end_fn(extra_in):
        return extra_in

    dynamic_slice_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([5]), start_fn, end_fn, step)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_dynamic_slice_one_dim_mix_up(step):
    def start_fn(extra_in):
        tmp = extra_in + 3
        tmp = tmp - 10
        tmp = tmp + 3

        return tmp

    def end_fn(extra_in):
        tmp = extra_in - 6
        tmp = tmp + 4
        return tmp

    dynamic_slice_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([5]), start_fn, end_fn, step)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_dynamic_slice_two_dims(step):
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 1

    dynamic_slice_harness(
        torch.tensor([[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                      [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
        torch.tensor([0]), start_fn, end_fn, step)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_dynamic_slice_two_dims_twice_sliced(step):
    start_dim_one = torch.tensor([1])
    start_dim_two = torch.tensor([0])

    op = lambda t: t[start_dim_one:start_dim_one + 2:step, start_dim_two:
                     start_dim_two + 4:step]

    tensor_in = torch.tensor([[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                              [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                              [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                              [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])

    model = helpers.ModelWithWeights(op, tensor_in.shape)

    # Run on CPU.
    native_out, _ = model((tensor_in, ))

    # Run on IPU.
    poptorch_model = poptorch.trainingModel(model)
    poptorch_out, _ = poptorch_model((tensor_in, ))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


def test_dynamic_slice_one_dim_equal():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in

    error_msg = r"The start and end of a slice must be different."

    with pytest.raises(RuntimeError, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_slice_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                              torch.tensor([5]),
                              start_fn,
                              end_fn,
                              1,
                              test_training=False)


def test_dynamic_slice_one_dim_less_than():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in - 2

    error_msg = (r"Taking a slice of a tensor with the end less than the " +
                 r"start is not supported.")

    with pytest.raises(RuntimeError, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_slice_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                              torch.tensor([5]),
                              start_fn,
                              end_fn,
                              2,
                              test_training=False)


def test_dynamic_slice_one_dim_multiply():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in * 2

    error_msg = (
        r"The size of the sliced tensor must be a constant for each " +
        r"execution of the model when running on the IPU\.")

    with pytest.raises(RuntimeError, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_slice_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                              torch.tensor([5]),
                              start_fn,
                              end_fn,
                              3,
                              test_training=False)


def test_dynamic_slice_one_dim_add_non_factor():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 7

    error_msg = (r"The size of the slice \(7\) must be a factor of the " +
                 r"slicing dimension \(8\)\.")

    with pytest.raises(RuntimeError, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_slice_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                              torch.tensor([1]),
                              start_fn,
                              end_fn,
                              1,
                              test_training=False)


def test_dynamic_slice_one_dim_mix_up_float():
    def start_fn(extra_in):
        tmp = extra_in + 3
        tmp = tmp - 10.5
        tmp = tmp + 3.5

        return tmp.to(torch.int32)

    def end_fn(extra_in):
        tmp = extra_in - 6.5
        tmp = tmp + 4.5
        return tmp.to(torch.int32)

    error_msg = (
        r"The size of the sliced tensor must be a constant for each " +
        r"execution of the model when running on the IPU\. In this case, " +
        r"there is a float added to the slice indices meaning it may change " +
        r"between runs\.")

    with pytest.raises(RuntimeError, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_slice_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                              torch.tensor([5]),
                              start_fn,
                              end_fn,
                              2,
                              test_training=False)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_unbind(dim):
    op = lambda x: torch.unbind(x, dim)
    x = torch.randn(2, 3, 4)
    model = helpers.ModelWithWeights(op, x.shape, out_fn=lambda x: x[0])
    poptorch_model = poptorch.trainingModel(model)

    native_out, _ = model((x, ))
    poptorch_out, _ = poptorch_model((x, ))

    # Check the unbound dim length is the same
    assert len(native_out) == len(poptorch_out)

    # Inference test - check outputs
    for tensor_native, tensor_pop in zip(native_out, poptorch_out):
        helpers.assert_allclose(expected=tensor_native, actual=tensor_pop)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()
