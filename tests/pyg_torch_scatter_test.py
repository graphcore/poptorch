#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Tests for PyG torch_scatter ops integration with PopTorch
from functools import partial
import torch
import pytest
import helpers
import poptorch

if helpers.is_running_tests:
    from torch_scatter import scatter, scatter_log_softmax, scatter_softmax, scatter_std, scatter_logsumexp, scatter_add, scatter_max, scatter_min, scatter_mul
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

    def scatter_min():
        pass

    def scatter_mul():
        pass

    def scatter_logsumexp():
        pass


def torch_scatter_harness(func, src, index, out=None):

    dim_size = int(index.max()) + 1

    class Model(torch.nn.Module):
        def forward(self, src, index, out=None):
            if out is None:
                return func(src, index, dim_size=dim_size)
            return func(src, index, out=out, dim_size=dim_size)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    out_in_plac_native = None

    if out is not None:
        out_in_plac_native = out.clone()
        native_out = func(src,
                          index,
                          out=out_in_plac_native,
                          dim_size=dim_size)
        ipu_out = poptorch_model(src, index, out=out)
    else:
        native_out = func(src, index, dim_size=dim_size)
        ipu_out = poptorch_model(src, index)

    helpers.assert_allclose(actual=ipu_out, expected=native_out)
    if out is not None:
        helpers.assert_allclose(actual=out, expected=out_in_plac_native)

    poptorch_model.destroy()


@pytest.mark.parametrize("reduce", ['sum', 'mean', 'max', 'min', 'mul'])
def test_scatter(reduce):
    func = partial(scatter, reduce=reduce)
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(func, src, index)


@pytest.mark.parametrize(
    "func",
    [scatter_log_softmax, scatter_logsumexp, scatter_softmax, scatter_std])
def test_composites(func):
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 5, 3]).long()
    torch_scatter_harness(func, src, index)


@pytest.mark.parametrize("func", [scatter_max, scatter_min, scatter_mul])
def test_scatter_inplace(func):
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 4, 2, 3, 5]).long()
    out = torch.tensor([10, 1, 11, 1, 23, 1]).float()
    torch_scatter_harness(func, src, index, out)


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
@pytest.mark.parametrize("func", [scatter_max, scatter_min, scatter_mul])
def test_scatter_overloads(shape, func):
    torch.manual_seed(0)
    x = torch.rand(shape)
    ind = torch.randint(3, shape)

    torch_scatter_harness(func, x, ind)
