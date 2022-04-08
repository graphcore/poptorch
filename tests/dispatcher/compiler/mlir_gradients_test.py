#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import torch
import pytest
import helpers
from poptorch.experimental import IPUContext


@pytest.mark.mlirSupportRequired
def test_grad():
    model = torch.nn.Sequential(torch.nn.Linear(1, 10))

    t1 = torch.tensor([3.5])
    t2 = torch.ones([10])

    def grad(x1, x2, lin):
        out = lin(x1)
        loss = torch.nn.functional.mse_loss(out, x2)
        out.retain_grad()
        loss.backward()
        return loss, lin[0].weight.grad, lin[0].bias.grad, out.grad

    cpu_model = copy.deepcopy(model)
    cpu_result = grad(t1, t2, cpu_model)

    ipu_result = IPUContext(grad,
                            parameters_and_buffers=model.named_parameters())(
                                t1, t2, model)

    for ipu_out, cpu_out in zip(ipu_result, cpu_result):
        helpers.assert_allclose(expected=cpu_out, actual=ipu_out)


@pytest.mark.mlirSupportRequired
def test_SGD():
    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(1, 10))
    cpu_model = copy.deepcopy(model)
    t1 = torch.tensor([3.5])
    t2 = torch.ones([10])

    ipu_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cpu_opt = torch.optim.SGD(cpu_model.parameters(), lr=0.01)

    def sgd(x1, x2, m, o):
        o.zero_grad()
        out = m(x1)

        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()

        o.step()
        return m[0].weight.grad, m[0].bias.grad, m[0].weight, m[0].bias

    ipu_sgd = IPUContext(sgd, parameters_and_buffers=model.named_parameters())

    for _ in range(5):
        # Run on IPU.
        ipu_result = ipu_sgd(t1, t2, model, ipu_opt)

        cpu_result = sgd(t1, t2, cpu_model, cpu_opt)

        for ipu_out, cpu_out in zip(ipu_result, cpu_result):
            helpers.assert_allclose(expected=ipu_out, actual=cpu_out)


@pytest.mark.mlirSupportRequired
def test_Adam():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(1, 10))
    cpu_model = copy.deepcopy(model)

    t1 = torch.tensor([3.5])
    t2 = torch.ones([10])

    ipu_opt = torch.optim.Adam(model.parameters(), lr=0.01)
    cpu_opt = torch.optim.Adam(cpu_model.parameters(), lr=0.01)

    def adam(x1, x2, m, o):
        o.zero_grad()
        out = m(x1)

        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()

        o.step()
        return m[0].weight.grad, m[0].bias.grad, m[0].weight, m[0].bias

    ipu_adam = IPUContext(adam,
                          parameters_and_buffers=model.named_parameters())

    for i in range(0, 5):
        # Run on IPU.
        ipu_result = ipu_adam(t1, t2, model, ipu_opt)

        cpu_result = adam(t1, t2, cpu_model, cpu_opt)

        for ipu_out, cpu_out in zip(ipu_result, cpu_result):
            if i == 0:
                # Until we can detect changes to and update in flight some "constants",
                # Adam will diverge quite quickly. First step should be 1:1.
                helpers.assert_allclose(expected=ipu_out, actual=cpu_out)
            else:
                helpers.assert_allclose(expected=ipu_out,
                                        actual=cpu_out,
                                        atol=0.01,
                                        rtol=1e-03)
