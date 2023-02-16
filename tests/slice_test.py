#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import copy

import pytest
import torch

import helpers
import poptorch


def slice_test_harness(tensor_x, tensor_y, start_fn, end_fn, step):
    op = lambda x, y: x[start_fn(x):end_fn(x):step] + y

    model = helpers.ModelWithWeights(op, tensor_x.shape)

    # Run on CPU.
    native_out, _ = model((tensor_x, tensor_y))

    # Run on IPU.
    options = poptorch.Options()
    poptorch_model = poptorch.trainingModel(model, options=options)
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


def dynamic_slice_harness(tensor_in,
                          extra_in,
                          start_fn,
                          end_fn,
                          step,
                          test_training=False):
    # TODO(T62094) PopART doesn't currently support dynamic slices in training.
    # Once it works, switch back test_training to True by default.
    options = poptorch.Options()
    if test_training:
        size = end_fn(1) - start_fn(1)
        op = lambda t, e: poptorch.dynamic_slice(t, 0, start_fn(e), size, step)
        model = helpers.ModelWithWeights(op, tensor_in.shape)

        # Run on CPU.
        native_out, _ = model((tensor_in, extra_in))

        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model, options)
        poptorch_out, _ = poptorch_model((tensor_in, extra_in))

        # Training test - check weights changed
        poptorch_model.assert_weights_changed()
    else:
        model = torch.nn.Module()
        size = (end_fn(torch.tensor([1], dtype=torch.int)) -
                start_fn(torch.tensor([1], dtype=torch.int))).item()
        model.forward = lambda t, e: poptorch.dynamic_slice(
            t, 0, start_fn(e), size, step)

        # Run on CPU.
        native_out = model(tensor_in, extra_in)

        # Run on IPU.
        poptorch_model = poptorch.inferenceModel(model, options)
        # Make sure the model is compiled using different tensor values
        # otherwise there is no way to tell if the values are compiled
        # in the executable or truly dynamic.
        poptorch_model.compile(
            torch.randn_like(tensor_in),  # Use a random input
            extra_in + torch.tensor([20])  # Offset extra_in
        )
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
    options = poptorch.Options()
    poptorch_model = poptorch.trainingModel(model, options=options)
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

    with pytest.raises(poptorch.Error, match=error_msg):
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

    error_msg = (r"Taking a slice of a tensor with the end less than the "
                 r"start is not supported.")

    with pytest.raises(poptorch.Error, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_slice_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                              torch.tensor([5]),
                              start_fn,
                              end_fn,
                              2,
                              test_training=False)


def test_dynamic_slice_one_dim_add_non_factor():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 7

    error_msg = (r"The size of the slice \(7\) must be a factor of the "
                 r"slicing dimension \(8\)\.")

    with pytest.raises(poptorch.Error, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_slice_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                              torch.tensor([1]),
                              start_fn,
                              end_fn,
                              1,
                              test_training=False)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("use_half", [True, False])
def test_unbind(dim, use_half):
    if use_half:
        # Test correct implicit casting
        def op(x):
            unbound = torch.unbind(x, dim)
            return unbound[0] + 2.0, unbound[1]
    else:
        op = lambda x: torch.unbind(x, dim)

    x = torch.randn(2, 3, 4)

    model = helpers.ModelWithWeights(op, x.shape, out_fn=lambda x: x[0])

    # Unfortunately not all forms of matmul are supported for torch.half on the
    # CPU (including 1-dim input, 2-dim weights), so we can only run the IPU
    # model with halves.
    poptorch_model = copy.deepcopy(model)
    if use_half:
        poptorch_model.half()
        # pylint: disable=protected-access
        poptorch_model._weights_before = poptorch_model.lin.weight.detach(
        ).clone()

    options = poptorch.Options()
    poptorch_model = poptorch.trainingModel(poptorch_model, options=options)

    native_out, _ = model((x, ))
    poptorch_out, _ = poptorch_model((x.half() if use_half else x, ))

    # Check the unbound dim length is the same
    assert len(native_out) == len(poptorch_out)

    # Inference test - check outputs
    for tensor_native, tensor_pop in zip(native_out, poptorch_out):
        if use_half:
            tensor_native = tensor_native.half()
        helpers.assert_allclose(expected=tensor_native,
                                actual=tensor_pop,
                                atol=0.01,
                                rtol=0.01)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


def test_scalarslice():
    class Model(torch.nn.Module):
        def forward(self, x):
            return (x / 2)[:]

    model = Model()
    options = poptorch.Options()
    poptorch_model = poptorch.inferenceModel(model, options)

    input_tensor = torch.tensor([2])
    assert poptorch_model(input_tensor) == model(input_tensor)


def test_select_negative_dim():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.select(-1, 1)

    model = Model()
    options = poptorch.Options()
    poptorch_model = poptorch.inferenceModel(model, options)

    input_tensor = torch.rand((2, 4))
    helpers.assert_allequal(actual=poptorch_model(input_tensor),
                            expected=model(input_tensor))


def test_slice_negative_dim():
    class Model(torch.nn.Module):
        def forward(self, x):
            # This lowers to aten::select with a negative dim, which is what
            # we want to test in the JIT dispatcher
            return x.narrow(-1, 0, 2)

    model = Model()
    options = poptorch.Options()
    poptorch_model = poptorch.inferenceModel(model, options)

    input_tensor = torch.rand((2, 4))
    helpers.assert_allequal(actual=poptorch_model(input_tensor),
                            expected=model(input_tensor))


def dynamic_update_harness(tensor_in,
                           src_in,
                           extra_in,
                           start_fn,
                           end_fn,
                           dim=0,
                           test_training=False):
    # TODO(T62094) PopART doesn't currently support dynamic slices in training.
    # Once it works, switch back test_training to True by default.
    options = poptorch.Options()
    if test_training:
        size = end_fn(1) - start_fn(1)
        op = lambda t, s, e: poptorch.dynamic_update(t, s, dim, start_fn(e),
                                                     size)
        model = helpers.ModelWithWeights(op, tensor_in.shape)

        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model, options)
        poptorch_out, _ = poptorch_model((tensor_in, src_in, extra_in))

        # Run on CPU.
        native_out, _ = model((tensor_in, src_in, extra_in))

        # Training test - check weights changed
        poptorch_model.assert_weights_changed()
    else:
        model = torch.nn.Module()
        size = (end_fn(torch.tensor([1], dtype=torch.int)) -
                start_fn(torch.tensor([1], dtype=torch.int))).item()
        model.forward = lambda t, s, e: poptorch.dynamic_update(
            t, s, dim, start_fn(e), size)

        # Run on IPU.
        poptorch_model = poptorch.inferenceModel(model, options)
        # Make sure the model is compiled using different tensor values
        # otherwise there is no way to tell if the values are compiled
        # in the executable or truly dynamic.
        poptorch_model.compile(
            torch.randn_like(tensor_in),  # Use a random input
            torch.randn_like(src_in),  # Use random source values
            extra_in + torch.tensor([20])  # Offset extra_in
        )
        poptorch_out = poptorch_model(tensor_in, src_in, extra_in)

        # Run on CPU.
        native_out = model(tensor_in, src_in, extra_in)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_dynamic_update_single_update():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 1

    dynamic_update_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([-1.0]), torch.tensor([1]), start_fn, end_fn)


def test_dynamic_update_one_dim_add():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 4

    dynamic_update_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([-1.0, -1.0, -1.0, -1.0]), torch.tensor([1]), start_fn,
        end_fn)


def test_dynamic_update_one_dim_subtract():
    def start_fn(extra_in):
        return extra_in - 4

    def end_fn(extra_in):
        return extra_in

    dynamic_update_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([-1.0, -1.0, -1.0, -1.0]), torch.tensor([5]), start_fn,
        end_fn)


def test_dynamic_update_one_dim_equal():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in

    error_msg = r"The start and end of a slice must be different"

    with pytest.raises(poptorch.Error, match=error_msg):
        dynamic_update_harness(
            torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            torch.tensor([-1.0]), torch.tensor([1]), start_fn, end_fn)


def test_dynamic_update_one_dim_add_non_factor():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 2

    # Set test_training=False because we expect inference to fail
    dynamic_update_harness(torch.tensor([2.0, 2.0, 3.0]),
                           torch.tensor([-1.0, -1.0]),
                           torch.tensor([1]),
                           start_fn,
                           end_fn,
                           test_training=False)


def test_dynamic_update_one_dim_less_than():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in - 2

    error_msg = (r"Taking a slice of a tensor with the end less than the "
                 r"start is not supported.")

    with pytest.raises(poptorch.Error, match=error_msg):
        # Set test_training=False because we expect inference to fail
        dynamic_update_harness(torch.tensor(
            [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                               torch.tensor([7.0, 8.0]),
                               torch.tensor([5]),
                               start_fn,
                               end_fn,
                               test_training=False)


def test_dynamic_update_two_dims():
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 1

    dynamic_update_harness(
        torch.tensor([[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                      [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
        torch.tensor([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
        torch.tensor([0]), start_fn, end_fn)


def test_dynamic_update_wrong_dim():
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 1

    error_msg = (r"input and src tensors must have same dimensionality. "
                 r"\(2\) vs \(1\)")

    with pytest.raises(poptorch.Error, match=error_msg):
        dynamic_update_harness(
            torch.tensor([[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
            torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            torch.tensor([0]), start_fn, end_fn)


def test_dynamic_update_two_dims_dim1():
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 1

    dynamic_update_harness(torch.tensor(
        [[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
         [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
                           torch.tensor([[-1.0], [-1.0]]),
                           torch.tensor([4]),
                           start_fn,
                           end_fn,
                           dim=1)


def test_dynamic_update_3_dims_dim0():
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 2

    input = torch.ones(3, 4, 5)
    src = torch.ones(2, 4, 5) * -1.0

    dynamic_update_harness(input,
                           src,
                           torch.tensor([1]),
                           start_fn,
                           end_fn,
                           dim=0)


def test_dynamic_update_3_dims_dim1():
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 2

    input = torch.ones(3, 4, 5)
    src = torch.ones(3, 2, 5) * -1.0

    dynamic_update_harness(input,
                           src,
                           torch.tensor([1]),
                           start_fn,
                           end_fn,
                           dim=1)


def test_dynamic_update_3_dims_dim2():
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 3

    input = torch.ones(3, 4, 5)
    src = torch.ones(3, 4, 3) * -1.0

    dynamic_update_harness(input,
                           src,
                           torch.tensor([2]),
                           start_fn,
                           end_fn,
                           dim=2)


def test_dynamic_update_wrong_dtype():
    t = torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    s = torch.tensor([-1])
    idx = torch.tensor([1])
    model = torch.nn.Module()
    model.forward = lambda t, s, e: poptorch.dynamic_update(t, s, 0, idx, 1)

    # Run on IPU.
    options = poptorch.Options()
    poptorch_model = poptorch.inferenceModel(model, options)

    error_msg = (r"input and src tensor must have same dtype\."
                 r" \(torch\.float32 vs torch.int32\)")

    with pytest.raises(poptorch.Error, match=error_msg):
        poptorch_model.compile(
            torch.randn_like(t),  # Use a random input
            s,
            idx + torch.tensor([20])  # Offset extra_in
        )
