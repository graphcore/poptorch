#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Tests for PyG torch_scatter ops integration with PopTorch
from functools import partial
import torch
import pytest
import helpers
import poptorch

if helpers.is_running_tests:
    from torch_scatter import scatter, scatter_log_softmax, scatter_softmax, scatter_std, scatter_add
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


def torch_scatter_harness(trace_model, func, src, index):
    class Model(torch.nn.Module):
        def forward(self, src, index):
            return func(src, index)

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options=options)
    native_out = func(src, index)
    ipu_out = poptorch_model(src, index)
    helpers.assert_allclose(actual=ipu_out, expected=native_out)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("reduce", ['sum', 'mean', 'max', 'min'])
def test_scatter(trace_model, reduce):
    if not trace_model:
        pytest.skip("TODO(T65186): Various failures with the dispatcher.")

    func = partial(scatter, reduce=reduce)
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(trace_model, func, src, index)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("func",
                         [scatter_log_softmax, scatter_softmax, scatter_std])
def test_composites(trace_model, func):
    if not trace_model:
        pytest.skip("TODO(T65186): Various failures with the dispatcher.")

    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(trace_model, func, src, index)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_scatter_add_zeros_optimized(capfd):
    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(True, scatter_add, src, index)

    it = helpers.LogChecker(capfd).createIterator()
    it.findNext("Removing zeros output to scatter_add")


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_scatter_add_nd_expand_removed(capfd):
    torch.manual_seed(0)
    src = torch.randn(10, 6, 16)
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    func = partial(scatter_add, dim=1)
    torch_scatter_harness(True, func, src, index)

    it = helpers.LogChecker(capfd).createIterator()
    it.findNext("Removing index expansion node:")