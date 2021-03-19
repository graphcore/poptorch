#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@helpers.printCapfdOnExit
def test_autocast_decorator(capfd, setting):
    class ModelDefault(torch.nn.Module):
        @poptorch.autocast()
        def forward(self, x, y):
            return torch.bmm(x, y)

    class ModelTrue(torch.nn.Module):
        @poptorch.autocast(True)
        def forward(self, x, y):
            return torch.bmm(x, y)

    class ModelFalse(torch.nn.Module):
        @poptorch.autocast(False)
        def forward(self, x, y):
            return torch.bmm(x, y)

    if setting == "true":
        model = ModelTrue()
    elif setting == "false":
        model = ModelFalse()
    else:
        model = ModelDefault()

    torch.manual_seed(42)
    x = torch.randn(1, 20, 20)
    y = torch.randn(1, 20, 20)
    poptorch.setLogLevel('TRACE')
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x, y)

    testlog = helpers.LogChecker(capfd)
    if setting == "false":
        testlog.assert_contains(
            "Float(1:400, 20:20, 20:1, requires_grad=0, device=cpu)" +
            " = popart::matmul")
    else:
        testlog.assert_contains(
            "Half(1:400, 20:20, 20:1, requires_grad=0, device=cpu)" +
            " = popart::matmul")


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@helpers.printCapfdOnExit
def test_autocast_block(capfd, setting):
    class ModelDefault(torch.nn.Module):
        def forward(self, x, y):
            with poptorch.autocast():
                return torch.bmm(x, y)

    class ModelTrue(torch.nn.Module):
        def forward(self, x, y):
            with poptorch.autocast(True):
                return torch.bmm(x, y)

    class ModelFalse(torch.nn.Module):
        def forward(self, x, y):
            with poptorch.autocast(False):
                return torch.bmm(x, y)

    if setting == "true":
        model = ModelTrue()
    elif setting == "false":
        model = ModelFalse()
    else:
        model = ModelDefault()

    torch.manual_seed(42)
    x = torch.randn(1, 20, 20)
    y = torch.randn(1, 20, 20)
    poptorch.setLogLevel('TRACE')
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x, y)

    testlog = helpers.LogChecker(capfd)
    if setting == "false":
        testlog.assert_contains(
            "Float(1:400, 20:20, 20:1, requires_grad=0, device=cpu)" +
            " = popart::matmul")
    else:
        testlog.assert_contains(
            "Half(1:400, 20:20, 20:1, requires_grad=0, device=cpu)" +
            " = popart::matmul")


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@helpers.printCapfdOnExit
def test_enable_autocast(capfd, setting):
    torch.manual_seed(42)
    x = torch.randn(1, 1, 20, 20)
    model = torch.nn.Conv2d(1, 20, 5)
    model.autocast()

    opts = poptorch.Options()
    if setting == "true":
        opts.Precision.autocastEnabled(True)
    elif setting == "false":
        opts.Precision.autocastEnabled(False)

    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch.setLogLevel('TRACE')
    poptorch_model(x)

    testlog = helpers.LogChecker(capfd)
    if setting == "false":
        testlog.assert_contains(
            "Float(1:5120, 20:256, 16:16, 16:1, requires_grad=1, device=cpu)" +
            " = popart::conv")
    else:
        testlog.assert_contains(
            "Half(1:5120, 20:256, 16:16, 16:1, requires_grad=1, device=cpu)" +
            " = popart::conv")


@pytest.mark.parametrize("setting", {"hff", "hfh", "hhf", "default"})
@helpers.printCapfdOnExit
def test_autocast_policy(capfd, setting):
    class PolicyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 20, 20)

        def forward(self, x):
            return torch.relu(0.5 * self.conv.forward(x))

    torch.manual_seed(42)
    x = torch.randn(1, 1, 20, 20)
    model = PolicyModel()
    model.autocast()

    if setting == "hff":
        policy = poptorch.autocasting.Policy([torch.nn.Conv2d],
                                             [torch.mul, torch.relu])
    elif setting == "hfh":
        policy = poptorch.autocasting.Policy([torch.nn.Conv2d, torch.relu],
                                             [torch.mul])
    elif setting == "hhf":
        policy = poptorch.autocasting.Policy([torch.mul],
                                             [torch.nn.Conv2d, torch.relu])
    else:
        policy = poptorch.autocasting.Policy()

    poptorch.setLogLevel('TRACE')
    opts = poptorch.Options()
    opts.Precision.autocastPolicy(policy)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x)

    testlog = helpers.LogChecker(capfd)
    if setting == "hff":
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::conv")
        testlog.assert_contains(
            "Float(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::mul")
        testlog.assert_contains(
            "Float(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::relu")
    elif setting == "hfh":
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::conv")
        testlog.assert_contains(
            "Float(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::mul")
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::relu")
    elif setting == "hhf":
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::conv")
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::mul")
        testlog.assert_contains(
            "Float(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::relu")
    else:
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::conv")
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::mul")
        testlog.assert_contains(
            "Half(1:1, 1:1, 1:1, 1:1, requires_grad=1, device=cpu)" +
            " = popart::relu")
