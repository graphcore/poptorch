#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch
import helpers


# Reduce Ops Harness
# Checks that the IPU reduce ops match the CPU version.
def reduce_harness(trace_model,
                   func,
                   input,
                   dim=None,
                   expected_dtype=torch.bool):
    # dim must be passed this way to avoid named tensor errors
    kwargs = {"dim": dim} if dim else {}

    def reduce_op(x):
        return func(x, **kwargs)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.reduce_op = reduce_op

        def forward(self, x):
            # Ensure input is not modified in place
            x = x + 0
            return self.reduce_op(x)

    model = Model()

    # Run on IPU and check that the result has the correct type
    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)
    pop_model = poptorch.inferenceModel(model, opts)
    pop_out = pop_model(input)
    assert pop_out.dtype == expected_dtype

    native_out = model(input)
    assert native_out.size() == pop_out.size()

    if torch.is_floating_point(native_out):
        helpers.assert_allclose(expected=native_out, actual=pop_out)
    else:
        helpers.assert_allequal(expected=native_out, actual=pop_out)


# torch.all, torch.any
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("func", [torch.all, torch.any])
@pytest.mark.parametrize("trace_model", [True, False])
def test_any_all(trace_model, func, dim):
    input = torch.randint(low=0, high=3, size=(32, 128))
    reduce_harness(trace_model, func, input, dim)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("func", [torch.sum, torch.mean])
@pytest.mark.parametrize("trace_model", [True, False])
def test_sum_mean(trace_model, func, dim):
    input = torch.rand(32, 128)
    reduce_harness(trace_model, func, input, dim, expected_dtype=torch.float32)
