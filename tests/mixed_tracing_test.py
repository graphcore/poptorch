#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import helpers
import poptorch


# Test that JIT tracing and the dispatcher can be used one after the other.
def test_mix_traced_dispatched():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cpu = poptorch.CPU(self.foo, "MyCPUOp")

        def foo(self, x, y):
            return x * y

        def forward(self, x, y):
            w = self.cpu(x, y)
            return w * 3.0

    in1 = torch.randn([5, 2, 3, 5])
    in2 = torch.tensor([2.0])

    model = Model()

    # Run with dispatcher
    dispatch_options = poptorch.Options()
    dispatch_options.Jit.traceModel(False)
    dispatched_model = poptorch.inferenceModel(model, dispatch_options)

    dispatched_output = dispatched_model(in1, in2)

    # Run with JIT trace
    traced_options = poptorch.Options()
    traced_options.Jit.traceModel(True)
    traced_model = poptorch.inferenceModel(model, traced_options)

    traced_output = traced_model(in1, in2)

    # Basic comparison
    helpers.assert_allclose(actual=dispatched_output, expected=traced_output)


# Just test that the dispatcher is disabled in the CPU op, and re-enabled
# afterwards.
def test_poptorch_op_in_cpu_op():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cpu = poptorch.CPU(self.foo, "MyCPUOp")

        def foo(self, x):
            return poptorch.identity_loss(x, reduction='sum')

        def forward(self, x):
            w = self.cpu(x)
            return w, self.foo(x)

    options = poptorch.Options()
    options.deviceIterations(2)

    options.Jit.traceModel(False)
    dispatched_model = poptorch.inferenceModel(Model(), options)

    # Just check it doesn't crash
    dispatched_model(torch.tensor([1.0, 2.0]))
