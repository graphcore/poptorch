#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch
import helpers

# Not need for mean or logsumexp
reduce_ops = [torch.sum, torch.prod]
test_tensors = [
    torch.tensor([1.0, 2.0, 3.1]),
    torch.tensor([1.1, 2.0, 3.0]),
    torch.tensor([0.0, 0.0, 0.0])
]


@pytest.mark.parametrize("op", reduce_ops)
@pytest.mark.parametrize("t_1", test_tensors)
@pytest.mark.parametrize("t_2", test_tensors)
@pytest.mark.parametrize("trace_model", [True, False])
def test_reduce_two_bool_types(op, t_1, t_2, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return op(x == y)

    model = Model()

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    native_out = model(t_1, t_2)
    poptorch_out = poptorch_model(t_1, t_2)
    #expected = no dims (scalar)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    assert native_out.dtype == torch.int64
    assert poptorch_out.dtype == torch.int32


@pytest.mark.parametrize("trace_model", [True, False])
def test_logits(trace_model):
    class Model(torch.nn.Module):
        def forward(self, logits, y):
            acc = torch.sum(torch.argmax(logits, -1) == y) / float(y.size(0))
            return acc

    model = Model()

    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    y = torch.tensor([[0], [2], [1]])

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    native_out = model(logits, y)
    poptorch_out = poptorch_model(logits, y)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
