#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers


@pytest.mark.parametrize("trace_model", [True, False])
def test_simple_CPU(trace_model):
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

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)

    model = Model()
    inference_model = poptorch.inferenceModel(model, options)

    in1 = torch.randn([5, 2, 3, 5])
    in2 = torch.tensor([2.0])

    out = inference_model(in1, in2)

    helpers.assert_allclose(actual=out, expected=in1 * 6.0, equal_nan=True)

    in2 = torch.tensor([4.0])

    out = inference_model(in1, in2)

    helpers.assert_allclose(actual=out, expected=in1 * 12.0, equal_nan=True)


@pytest.mark.parametrize("trace_model", [True, False])
def test_simple_CPU_multiple_outputs(trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cpu = poptorch.CPU(self.foo, "MyCPUOp")

        def foo(self, x, y):
            return x * y, x + y

        def forward(self, x, y):
            w, z = self.cpu(x, y)
            return w * 3.0, z

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)

    model = Model()
    inference_model = poptorch.inferenceModel(model, options)

    in1 = torch.randn([5, 2, 3, 5])
    in2 = torch.tensor([2.0])

    out, out2 = inference_model(in1, in2)

    helpers.assert_allclose(actual=out, expected=in1 * 6.0, equal_nan=True)

    helpers.assert_allclose(actual=out2, expected=(in1 + in2), equal_nan=True)

    in2 = torch.tensor([4.0])

    out, out2 = inference_model(in1, in2)

    helpers.assert_allclose(actual=out, expected=in1 * 12.0, equal_nan=True)
    helpers.assert_allclose(actual=out2, expected=(in1 + in2), equal_nan=True)


@pytest.mark.parametrize("trace_model", [True, False])
def test_CPU_reduce(trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cpu = poptorch.CPU(self.foo, "MyCPUOp")

        def foo(self, x):
            return torch.mean(x)

        def forward(self, x):
            w = self.cpu(x)
            return w

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)

    model = Model()
    inference_model = poptorch.inferenceModel(model, options)

    in1 = torch.randn([5, 2, 3, 5])
    out = inference_model(in1)

    helpers.assert_allclose(actual=out,
                            expected=torch.mean(in1),
                            equal_nan=True)


@pytest.mark.parametrize("trace_model", [True, False])
def test_CPU_matmul(trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.matmul = [torch.nn.Linear(20, 30)]

            self.cpu = poptorch.CPU(self.matmul[0], "MatMulOnCPU")

        def forward(self, input):
            return self.cpu(input)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = Model()
    inference_model = poptorch.inferenceModel(model, options)

    input = torch.randn(128, 20)
    out = inference_model(input)

    helpers.assert_allclose(actual=out,
                            expected=model.matmul[0](input),
                            equal_nan=True)


@pytest.mark.parametrize("trace_model", [True, False])
def test_CPU_multiple_calls(trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cpu = poptorch.CPU(self.foo, "MyCPUOp")

        def foo(self, x):
            assert x.device.type == "cpu", x.device.type
            return x * 2.0

        def forward(self, x):
            out = self.cpu(x)
            out = self.cpu(out)
            out = self.cpu(out)
            return out

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)

    model = Model()
    inference_model = poptorch.inferenceModel(model, options)

    in1 = torch.randn([5, 2, 3, 5])
    out = inference_model(in1)

    helpers.assert_allclose(actual=out, expected=in1 * 8.0, equal_nan=True)


@pytest.mark.parametrize("trace_model", [True, False])
def test_CPU_multiple_calls_multiple_classes(trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cpu = poptorch.CPU(self.foo, "MyCPUOp")
            self.cpu2 = poptorch.CPU(self.bar, "MyCPUOp2")

        def foo(self, x):
            return x * 2.0

        def bar(self, x, y):
            return x + y

        def forward(self, x, y):
            out = self.cpu(x)
            out = self.cpu2(out, y)

            out = self.cpu(out)
            out = self.cpu2(out, y)

            out = self.cpu(out)
            out = self.cpu2(out, y)

            return out

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)

    model = Model()
    inference_model = poptorch.inferenceModel(model, options)

    in1 = torch.randn([5])
    in2 = torch.randn([5])

    out = inference_model(in1, in2)

    helpers.assert_allclose(actual=out,
                            expected=model(in1, in2),
                            equal_nan=True)
