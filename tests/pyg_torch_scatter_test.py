#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Tests for PyG torch_scatter ops integration with PopTorch
from functools import partial
import torch
import pytest
import helpers
import poptorch

if helpers.is_running_tests:
    from torch_scatter import scatter, scatter_log_softmax, scatter_softmax, scatter_std, scatter_add, scatter_max
else:

    def scatter():
        pass

    def scatter_log_softmax():
        pass

    def scatter_softmax():
        pass

    def scatter_std():
        pass

    def scatter_add():
        pass

    def scatter_max():
        pass


def torch_scatter_fusible_model(func, src, index, dtype):

    # We do the shape inference from scatter here because we don't support
    # dynamic shaped tensors on the ipu
    dim_size = int(index.max()) + 1

    class Model(torch.nn.Module):
        def forward(self, src, index, dtype):
            ones = torch.ones_like(src, dtype=dtype)
            two = torch.ones_like(src) * 2
            out = func(src, index, dim_size=dim_size)
            out_ones = func(ones, index, dim_size=dim_size)
            out_two = func(two, index, dim_size=dim_size)
            if isinstance(out, tuple):
                out = out[0]
                out_ones = out_ones[0]
                out_two = out_two[0]

            src_updated = src - torch.sum(out)
            out_updated, _ = scatter_max(src_updated, index, dim_size=dim_size)

            return (out_ones + out_two) / torch.sum(out_updated)

    model = Model()
    options = poptorch.Options()
    poptorch_model = poptorch.inferenceModel(model, options=options)

    ones = torch.ones_like(src, dtype=dtype)
    two = torch.ones_like(src) * 2

    native_out = func(src, index, dim_size=dim_size)
    native_out_ones = func(ones, index, dim_size=dim_size)
    native_out_two = func(two, index, dim_size=dim_size)
    if isinstance(native_out, tuple):
        native_out = native_out[0]
        native_out_ones = native_out_ones[0]
        native_out_two = native_out_two[0]

    src_updated = src - torch.sum(native_out)
    native_out_updated, _ = scatter_max(src_updated, index, dim_size=dim_size)

    expected_nat = (native_out_ones +
                    native_out_two) / torch.sum(native_out_updated)

    ipu_out = poptorch_model(src, index, dtype)

    helpers.assert_allclose(actual=torch.nan_to_num(ipu_out),
                            expected=torch.nan_to_num(expected_nat))


def torch_scatter_harness(func, src, index):

    dim_size = int(index.max()) + 1

    class Model(torch.nn.Module):
        def forward(self, src, index):
            return func(src, index, dim_size=dim_size)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    native_out = func(src, index, dim_size=dim_size)
    ipu_out = poptorch_model(src, index)
    helpers.assert_allclose(actual=ipu_out, expected=native_out)


@pytest.mark.parametrize("reduce", ['sum', 'mean', 'max', 'min'])
def test_scatter(reduce):
    func = partial(scatter, reduce=reduce)
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(func, src, index)


@pytest.mark.parametrize("func",
                         [scatter_log_softmax, scatter_softmax, scatter_std])
def test_composites(func):
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(func, src, index)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_scatter_add_zeros_optimized(capfd):
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(scatter_add, src, index)

    it = helpers.LogChecker(capfd).createIterator()
    it.findNext("Removing zeros output to scatter_add")


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_scatter_add_nd_expand_removed(capfd):
    torch.manual_seed(0)
    src = torch.randn(10, 6, 16)
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    func = partial(scatter_add, dim=1)
    torch_scatter_harness(func, src, index)

    it = helpers.LogChecker(capfd).createIterator()
    it.findNext("Removing index expansion node:")


@pytest.mark.parametrize("shape", [(5, ), (2, 5), (2, 5, 5)])
def test_scatter_max(shape):
    torch.manual_seed(0)
    x = torch.rand(shape)
    ind = torch.randint(3, shape)

    torch_scatter_harness(scatter_max, x, ind)


@pytest.mark.parametrize("shape", [(5, ), (2, 5), (2, 5, 5)])
@pytest.mark.parametrize("func", [
    scatter, scatter_add, scatter_max, scatter_softmax, scatter_log_softmax,
    scatter_std
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int])
def test_scatter_fuse(shape, func, dtype):
    if dtype != torch.float32 and func in [
            scatter_softmax, scatter_log_softmax, scatter_std
    ]:
        pytest.skip("can only be computed with fp32 data types")

    torch.manual_seed(0)
    x = torch.rand(shape)
    ind = torch.randint(3, shape)

    torch_scatter_fusible_model(func, x, ind, dtype)
