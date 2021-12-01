#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch.optim as optim
import poptorch
import helpers


def blas_op(op, input1, input2, out, trace_model):
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
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)
    if out is not None:
        helpers.assert_allclose(expected=native_out,
                                actual=out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)


@pytest.mark.parametrize("optional_out", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_matmul(optional_out, trace_model):
    torch.manual_seed(42)

    input1 = torch.randn([10, 200])
    input2 = torch.randn([200, 45])
    out = torch.randn([10, 45]) if optional_out else None

    blas_op(torch.matmul, input1, input2, out, trace_model)


@pytest.mark.parametrize("optional_out", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_bmm(optional_out, trace_model):
    input1 = torch.randn([12, 10, 200])
    input2 = torch.randn([12, 200, 33])
    out = torch.randn([12, 10, 33]) if optional_out else None

    blas_op(torch.bmm, input1, input2, out, trace_model)


@pytest.mark.parametrize("bias", [True, False])
def test_matmul_training(bias):
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

    optimizer = optim.SGD(model.parameters(), lr=0.01)
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
