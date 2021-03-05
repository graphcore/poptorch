#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch

import poptorch
import helpers


def slice_test_harness(tensor_x, tensor_y, start_fn, end_fn):
    class SliceModel(torch.nn.Module):
        def forward(self, x, y):
            return x[start_fn(x):end_fn(x)] + y

    model = SliceModel()

    # Run on CPU.
    native_out = model(tensor_x, tensor_y)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(tensor_x, tensor_y)

    helpers.assert_allequal(expected=native_out, actual=poptorch_out)


def test_slice_idx_size_of():
    def start_fn(tensor_in):
        return tensor_in.shape[0] // 2

    def end_fn(tensor_in):
        return tensor_in.shape[0] - 1

    slice_test_harness(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                       torch.tensor([3.0]), start_fn, end_fn)


def test_slice_with_sum():
    def start_fn(tensor_in):
        del tensor_in
        return torch.sum(torch.tensor([1, 2, 3])) // 3 - 2

    def end_fn(tensor_in):
        del tensor_in
        return torch.sum(torch.tensor([1, 2, 3])) // 3 + 1

    slice_test_harness(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                       torch.tensor([-3.0]), start_fn, end_fn)


def test_slice_with_branch():
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
                       torch.tensor([-3.0]), start_fn, end_fn)


def index_test_harness(model, inds):
    t = torch.arange(120).reshape(2, 3, 4, 5)
    inds_tensors = []
    for i in inds:
        inds_tensors.append(torch.tensor(i))
    inds_tuple = tuple(inds_tensors)
    poptorch_model = poptorch.inferenceModel(model)

    # Run on CPU
    native_out = model(t, *inds_tuple)
    # Run on IPU
    poptorch_out = poptorch_model(t, *inds_tuple)

    assert native_out.size() == poptorch_out.size()
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.parametrize(
    "inds",
    ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]]),
)
def test_index_1(inds):
    class Model(torch.nn.Module):
        def forward(self, t, idx):
            return t[idx]

    index_test_harness(Model(), inds)


@pytest.mark.parametrize(
    "inds",
    ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]]),
)
def test_index_2(inds):
    class Model(torch.nn.Module):
        def forward(self, t, idx):
            return t[idx, idx]

    index_test_harness(Model(), inds)


@pytest.mark.parametrize(
    "inds",
    ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]]),
)
def test_index_3(inds):
    class Model(torch.nn.Module):
        def forward(self, t, idx):
            return t[:, idx]

    index_test_harness(Model(), inds)


@pytest.mark.parametrize(
    "inds",
    ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]]),
)
def test_index_4(inds):
    class Model(torch.nn.Module):
        def forward(self, t, idx):
            return t[idx, :, idx]

    index_test_harness(Model(), inds)


@pytest.mark.parametrize(
    "inds",
    ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]]),
)
def test_index_5(inds):
    class Model(torch.nn.Module):
        def forward(self, t, idx):
            return t[:, :, idx]

    index_test_harness(Model(), inds)


@pytest.mark.parametrize(
    "inds",
    ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]]),
)
def test_index_6(inds):
    class Model(torch.nn.Module):
        def forward(self, t, idx):
            return t[:, idx, idx]

    index_test_harness(Model(), inds)


@pytest.mark.parametrize(
    "inds",
    ([[0]], [[1]], [[0, 1]], [[1, 0]], [[[0, 1], [1, 0]]]),
)
def test_index_7(inds):
    class Model(torch.nn.Module):
        def forward(self, t, idx):
            return t[:, idx, :, idx]

    index_test_harness(Model(), inds)


def dynamic_slice_harness(tensor_in, extra_in, start_fn, end_fn):
    class DynamicSliceModel(torch.nn.Module):
        def forward(self, tensor_in, extra_in):
            return tensor_in[start_fn(extra_in):end_fn(extra_in)]

    model = DynamicSliceModel()

    # Run on CPU.
    native_out = model(tensor_in, extra_in)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(tensor_in, extra_in)

    helpers.assert_allequal(expected=native_out, actual=poptorch_out)


def test_dynamic_slice_one_dim_add():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 4

    dynamic_slice_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([1]), start_fn, end_fn)


def test_dynamic_slice_one_dim_subtract():
    def start_fn(extra_in):
        return extra_in - 4

    def end_fn(extra_in):
        return extra_in

    dynamic_slice_harness(
        torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([5]), start_fn, end_fn)


def test_dynamic_slice_one_dim_mix_up():
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
        torch.tensor([5]), start_fn, end_fn)


def test_dynamic_slice_two_dims():
    def start_fn(extra_in):
        return extra_in.to(torch.int32)

    def end_fn(extra_in):
        return extra_in.to(torch.int32) + 1

    dynamic_slice_harness(
        torch.tensor([[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                      [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
        torch.tensor([0]), start_fn, end_fn)


def test_dynamic_slice_two_dims_twice_sliced():
    class Model(torch.nn.Module):
        def forward(self, tensor_in, start_dim_one, start_dim_two):
            return tensor_in[start_dim_one:start_dim_one +
                             2, start_dim_two:start_dim_two + 4]

    tensor_in = torch.tensor([[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                              [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                              [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                              [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    start_dim_one = torch.tensor([1])
    start_dim_two = torch.tensor([0])

    model = Model()

    # Run on CPU.
    native_out = model(tensor_in, start_dim_one, start_dim_two)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(tensor_in, start_dim_one, start_dim_two)
    helpers.assert_allequal(expected=native_out, actual=poptorch_out)


def test_dynamic_slice_one_dim_equal():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in

    error_msg = r"The start and end of a slice must be different."

    with pytest.raises(RuntimeError, match=error_msg):
        dynamic_slice_harness(
            torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            torch.tensor([5]), start_fn, end_fn)


def test_dynamic_slice_one_dim_less_than():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in - 2

    error_msg = (r"Taking a slice of a tensor with the end less than the " +
                 r"start is not supported.")

    with pytest.raises(RuntimeError, match=error_msg):
        dynamic_slice_harness(
            torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            torch.tensor([5]), start_fn, end_fn)


def test_dynamic_slice_one_dim_multiply():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in * 2

    error_msg = (
        r"The size of the sliced tensor must be a constant for each " +
        r"execution of the model when running on the IPU\.")

    with pytest.raises(RuntimeError, match=error_msg):
        dynamic_slice_harness(
            torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            torch.tensor([5]), start_fn, end_fn)


def test_dynamic_slice_one_dim_add_non_factor():
    def start_fn(extra_in):
        return extra_in

    def end_fn(extra_in):
        return extra_in + 7

    error_msg = (r"The size of the slice \(7\) must be a factor of the " +
                 r"slicing dimension \(8\)\.")

    with pytest.raises(RuntimeError, match=error_msg):
        dynamic_slice_harness(
            torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            torch.tensor([1]), start_fn, end_fn)


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
        dynamic_slice_harness(
            torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            torch.tensor([5]), start_fn, end_fn)
