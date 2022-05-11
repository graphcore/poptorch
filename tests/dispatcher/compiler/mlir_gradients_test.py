#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import torch
import pytest
import helpers
from poptorch.experimental import IPUContext


@pytest.mark.mlirSupportRequired
def test_grad():
    torch.manual_seed(42)
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

    ipu_result = IPUContext(grad, model=model)(t1, t2, model)

    for ipu_out, cpu_out in zip(ipu_result, cpu_result):
        helpers.assert_allclose(expected=cpu_out, actual=ipu_out)


@pytest.mark.mlirSupportRequired
def test_SGD():
    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(1, 10))
    cpu_model = copy.deepcopy(model)
    t1 = torch.tensor([3.5])
    t2 = torch.ones([10])

    ipu_opt = []
    cpu_opt = []

    def sgd(x1, x2, m, opts):
        if not opts:
            opts.append(torch.optim.SGD(m.parameters(), lr=0.01))

        opts[0].zero_grad()
        out = m(x1)

        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()

        opts[0].step()
        return m[0].weight.grad, m[0].bias.grad, m[0].weight, m[0].bias

    ipu_sgd = IPUContext(sgd, model=model)

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

    ipu_opt = []
    cpu_opt = []

    def adam(x1, x2, m, o):
        if not o:
            o.append(torch.optim.Adam(m.parameters(), lr=0.01))
        o[0].zero_grad()
        out = m(x1)

        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()

        o[0].step()
        return m[0].weight.grad, m[0].bias.grad, m[0].weight, m[0].bias

    ipu_adam = IPUContext(adam, model=model)

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
