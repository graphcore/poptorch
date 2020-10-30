#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch


def slice_test_harness(tensor_x, tensor_y, start_fn, end_fn):
    class SliceModel(torch.nn.Module):
        def forward(self, x, y):
            return x[start_fn(x):end_fn(x)] + y

    model = SliceModel()

    # Run on CPU.
    nativeOut = model(tensor_x, tensor_y)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(tensor_x, tensor_y)

    assert torch.equal(nativeOut, poptorch_out)


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
