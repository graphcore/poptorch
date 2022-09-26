#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers


def blas_op(op, input1, input2, out, trace_model, atol=1e-04, rtol=1e-04):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x, y, out=None):
            return self.op(x, y, out=out)

    model = Model(op)
    args = [input1, input2]
    if out is not None:
        args.append(out)
    # Run on CPU.
    native_out = model(*args)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(*args)

    helpers.assert_allclose(expected=native_out,
                            actual=poptorch_out,
                            atol=atol,
                            rtol=rtol,
                            equal_nan=True)
    if out is not None:
        helpers.assert_allclose(expected=native_out,
                                actual=out,
                                atol=atol,
                                rtol=rtol,
                                equal_nan=True)


@pytest.mark.parametrize("optional_out", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_matmul(optional_out, trace_model):
    torch.manual_seed(42)

    input1 = torch.randn([10, 200])
    input2 = torch.randn([200, 45])
    out = torch.randn([10, 45]) if optional_out else None

    blas_op(torch.matmul, input1, input2, out, trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("mode",
                         (poptorch.MatMulSerializationMode.InputChannels,
                          poptorch.MatMulSerializationMode.ReducingDim,
                          poptorch.MatMulSerializationMode.OutputChannels,
                          poptorch.MatMulSerializationMode.Disabled))
@pytest.mark.parametrize("factor", (2, 5, 10))
@pytest.mark.parametrize("keep_precision", [True, False])
def test_serializedMatMul(trace_model, mode, factor, keep_precision):
    torch.manual_seed(42)

    input1 = torch.rand(1, 10, 200)

    input2_dim = 45

    if mode == poptorch.MatMulSerializationMode.OutputChannels:
        # Ensure the value is a multiple of factor
        input2_dim = input2_dim // factor * factor

    input2 = torch.rand(200, input2_dim)

    def serialise_matmal_op(input, other, out):
        assert out is None
        return poptorch.serializedMatMul(input, other, mode, factor,
                                         keep_precision)

    if keep_precision:
        input1 = input1.half()
        input2 = input2.half()
        blas_op(serialise_matmal_op,
                input1,
                input2,
                None,
                trace_model,
                rtol=0.01,
                atol=0.05)
    else:
        blas_op(serialise_matmal_op, input1, input2, None, trace_model)


@pytest.mark.parametrize("optional_out", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_bmm(optional_out, trace_model):
    input1 = torch.randn([12, 10, 200])
    input2 = torch.randn([12, 200, 33])
    out = torch.randn([12, 10, 33]) if optional_out else None

    blas_op(torch.bmm, input1, input2, out, trace_model)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_matmul_training(bias, trace_model):
    N, M, K, C = 100, 9, 7, 5

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            torch.manual_seed(42)
            self.linear = torch.nn.Linear(K, K, bias=bias)
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.loss = torch.nn.L1Loss(reduction="mean")

        def forward(self, x, y, target):
            x = self.linear(x)
            x = torch.matmul(x, y)
            return x, self.loss(x, target)

    torch.manual_seed(42)
    model = Net()
    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(42)
    poptorch_model = poptorch.trainingModel(model, opts, optimizer)
    x = torch.randn(N, M, K)
    y = torch.randn(K, K)
    target = torch.empty(N, M, K, dtype=torch.long).random_(0, C)

    for _ in range(0, 400):
        optimizer.zero_grad()
        poptorch_out, poptorch_loss = poptorch_model(x, y, target)
        native_out, native_loss = model(x, y, target)
        native_loss.backward(retain_graph=True)
        optimizer.step()

    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_out,
                            rtol=1e-02,
                            atol=1e-02)
    helpers.assert_allclose(actual=poptorch_loss,
                            expected=native_loss,
                            rtol=1e-03,
                            atol=1e-03)


@pytest.mark.parametrize(
    "params",
    [
        # input_shape, beta, alpha
        ((3, 7), 1.0, 1.0),
        ((3, 1), 1.0, 0.75),
        ((1, 7), 0.75, 1.0),
        ((1), 0.75, 0.75),
    ])
@pytest.mark.parametrize("trace_model", [True, False])
def test_addmm(params, trace_model):
    torch.manual_seed(42)

    input_shape, beta, alpha = params

    t1 = torch.randn(input_shape)
    t2 = torch.randn(3, 5)
    t3 = torch.randn(5, 7)

    class AddmmModel(torch.nn.Module):
        def __init__(self, beta, alpha):
            super().__init__()
            self.beta = beta
            self.alpha = alpha

        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3, beta=self.beta, alpha=self.alpha)

    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    model = AddmmModel(beta, alpha)
    cpu_result = model(t1, t2, t3)
    ipu_result = poptorch.inferenceModel(model, opts)(t1, t2, t3)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)
