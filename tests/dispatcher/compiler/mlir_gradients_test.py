#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# pylint: disable=no-member
import copy
import torch
import pytest
import helpers
import poptorch


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_grad():
    model = torch.nn.Sequential(torch.nn.Linear(1, 10))

    t1 = torch.tensor([3.5])
    t2 = torch.ones([10])

    with poptorch.IPUScope([t1, t2],
                           model.named_parameters(),
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = model(t1)
        loss = torch.nn.functional.mse_loss(out, t2)
        out.retain_grad()
        loss.backward()

        ipu.outputs([loss, model[0].weight.grad, model[0].bias.grad, out.grad])

    ipu_outs = ipu(t1, t2)

    cpu_model = copy.deepcopy(model)
    out = cpu_model(t1)
    loss = torch.nn.functional.mse_loss(out, t2)

    out.retain_grad()
    loss.backward()

    cpu_outs = (loss, cpu_model[0].weight.grad, cpu_model[0].bias.grad,
                out.grad)

    for ipu_out, cpu_out in zip(ipu_outs, cpu_outs):
        helpers.assert_allclose(expected=ipu_out, actual=cpu_out)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_SGD():
    torch.manual_seed(42)

    model = torch.nn.Sequential(torch.nn.Linear(1, 10))
    t1 = torch.tensor([3.5])
    t2 = torch.ones([10])

    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    cpu_model = copy.deepcopy(model)

    with poptorch.IPUScope([t1, t2],
                           model.named_parameters(),
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        opt.zero_grad()
        out = model(t1)

        loss = torch.nn.functional.mse_loss(out, t2)
        loss.backward()

        opt.step()
        ipu.outputs([
            model[0].weight.grad, model[0].bias.grad, model[0].weight,
            model[0].bias
        ])

    opt = torch.optim.SGD(cpu_model.parameters(), lr=0.01)

    for _ in range(0, 5):
        # Run on IPU.
        ipu_outs = ipu(t1, t2)

        opt.zero_grad()

        out = cpu_model(t1)

        loss = torch.nn.functional.mse_loss(out, t2)

        loss.backward()
        opt.step()

        cpu_outs = (cpu_model[0].weight.grad, cpu_model[0].bias.grad,
                    cpu_model[0].weight, cpu_model[0].bias)

        for ipu_out, cpu_out in zip(ipu_outs, cpu_outs):
            helpers.assert_allclose(expected=ipu_out, actual=cpu_out)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_Adam():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(1, 10))
    t1 = torch.tensor([3.5])
    t2 = torch.ones([10])

    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    cpu_model = copy.deepcopy(model)

    with poptorch.IPUScope([t1, t2],
                           model.named_parameters(),
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        opt.zero_grad()
        out = model(t1)

        loss = torch.nn.functional.mse_loss(out, t2)
        loss.backward()

        opt.step()
        ipu.outputs([
            model[0].weight.grad, model[0].bias.grad, model[0].weight,
            model[0].bias
        ])

    opt = torch.optim.Adam(cpu_model.parameters(), lr=0.01)

    for i in range(0, 5):
        # Run on IPU.
        ipu_outs = ipu(t1, t2)

        opt.zero_grad()

        out = cpu_model(t1)

        loss = torch.nn.functional.mse_loss(out, t2)

        loss.backward()
        opt.step()

        cpu_outs = (cpu_model[0].weight.grad, cpu_model[0].bias.grad,
                    cpu_model[0].weight, cpu_model[0].bias)

        for ipu_out, cpu_out in zip(ipu_outs, cpu_outs):
            if i == 0:
                # Until we can detect changes to and update in flight some "constants",
                # Adam will diverge quite quickly. First step should be 1:1.
                helpers.assert_allclose(expected=ipu_out, actual=cpu_out)
            else:
                helpers.assert_allclose(expected=ipu_out,
                                        actual=cpu_out,
                                        atol=0.01,
                                        rtol=1e-03)
