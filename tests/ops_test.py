#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
import torch
import poptorch
import helpers


@pytest.mark.parametrize("trace_model", [True, False])
def test_print_tensor(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.ipu_print_tensor(x)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    m = poptorch.inferenceModel(Model(), options)
    m(torch.randn(5))


@pytest.mark.parametrize("trace_model", [True, False])
def test_print_tensor_with_title(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.ipu_print_tensor(x, "my_tensor")

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    m = poptorch.inferenceModel(Model(), options)
    m(torch.randn(5))


@pytest.mark.parametrize("trace_model", [True, False])
def test_nop(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.nop(x) * 2

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    m = poptorch.inferenceModel(Model(), options)
    m(torch.randn(5))


@pytest.mark.parametrize("trace_model", [True, False])
def test_name_scope(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            with poptorch.NameScope("NameScope"):
                return x + y

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    torch.manual_seed(42)
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    poptorch_model(x, y)

    ir = poptorch_model._debugGetPopartIR()  # pylint: disable=protected-access
    assert ir.find('"name":"NameScope/Add:InPlace"') != -1


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.parametrize("trace_model", [True, False])
def test_available_memory_last_op(capfd, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.matmul(x, x)
            return poptorch.set_available_memory(x, 0.3)

    input = torch.randn(10, 10)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    poptorch_model.compile(input)

    # Check the trace log to make sure set_available_memory isn't pruned
    # before it's lowered to PopART
    ir_before_popart_regex = \
    (r"Graph right before popart:\n"
     r".*\n"
     r".* popart::matmul.*\n"
     r".* poptorch::set_available_memory.*")

    log = helpers.LogChecker(capfd)
    log.assert_matches(ir_before_popart_regex, per_line=False)
