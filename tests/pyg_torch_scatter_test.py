#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Tests for PyG torch_scatter ops integration with PopTorch
import torch
import pytest
from torch_scatter import scatter, scatter_log_softmax, scatter_softmax, scatter_std

import poptorch
from helpers import assert_allequal


def torch_scatter_harness(trace_model, module, src, index):
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(module, options=options)
    native_out = module(src, index)
    ipu_out = poptorch_model(src, index)
    assert_allequal(actual=ipu_out, expected=native_out)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("reduce", ['sum', 'mean', 'max', 'min'])
def test_scatter(trace_model, reduce):
    if not trace_model:
        pytest.skip("TODO(T65186): Various failures with the dispatcher.")

    class Model(torch.nn.Module):
        def forward(self, src, index):
            return scatter(src=src, index=index, reduce=reduce, dim=0)

    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(trace_model, Model(), src, index)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("func",
                         [scatter_log_softmax, scatter_softmax, scatter_std])
def test_composites(trace_model, func):
    if not trace_model:
        pytest.skip("TODO(T65186): Various failures with the dispatcher.")

    class Model(torch.nn.Module):
        def forward(self, src, index):
            return func(src=src, index=index, dim=0)

    src = torch.tensor([1, 3, 2, 4, 5, 6]).float()
    index = torch.tensor([0, 1, 0, 1, 1, 3]).long()
    torch_scatter_harness(trace_model, Model(), src, index)
