#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
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
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x, y)

    testlog = helpers.LogChecker(capfd)
    test_type = "Float" if setting == "false" else "Half"
    testlog.assert_contains(
        f"{test_type}(1, 20, 20, strides=[400, 20, 1], requires_grad=0,"
        " device=cpu) = popart::matmul")


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
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
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x, y)

    testlog = helpers.LogChecker(capfd)
    test_type = "Float" if setting == "false" else "Half"
    testlog.assert_contains(
        f"{test_type}(1, 20, 20, strides=[400, 20, 1], requires_grad=0,"
        " device=cpu) = popart::matmul")


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
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
    poptorch_model(x)

    testlog = helpers.LogChecker(capfd)
    test_type = "Float" if setting == "false" else "Half"
    testlog.assert_contains(
        f"{test_type}(1, 20, 16, 16, strides=[5120, 256, 16, 1],"
        " requires_grad=1, device=cpu) = popart::conv")


@pytest.mark.parametrize("setting", {"hff", "hfh", "hhf", "default"})
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
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

    opts = poptorch.Options()
    opts.Precision.autocastPolicy(policy)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x)

    testlog = helpers.LogChecker(capfd)
    test_ops = ["conv", "mul", "relu"]
    test_types = []
    if setting == "default":
        test_types = ["Half", "Half", "Half"]
    else:
        for c in setting:
            if c == "f":
                test_types.append("Float")
            elif c == "h":
                test_types.append("Half")

    for i, op in enumerate(test_ops):
        testlog.assert_contains(
            f"{test_types[i]}(1, 1, 1, 1, strides=[1, 1, 1, 1],"
            f" requires_grad=1, device=cpu) = popart::{op}")
