#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch


def harness(setting, Model, args, trace_model):
    opts = poptorch.Options()
    if setting == "true":
        opts.Precision.enableFloatingPointExceptions(True)
    elif setting == "false":
        opts.Precision.enableFloatingPointExceptions(False)
    opts.Jit.traceModel(trace_model)

    poptorch_model = poptorch.inferenceModel(Model(), opts)

    if setting == "true":
        with pytest.raises(poptorch.Error):
            poptorch_model(*args)
    else:
        poptorch_model(*args)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("setting", {"default", "true", "false"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_div0(setting, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x / y

    x = torch.ones(10, 10)
    y = torch.zeros(10, 10)
    harness(setting, Model, [x, y], trace_model)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("setting", {"default", "true", "false"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_mul0inf(setting, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x * y

    x = torch.zeros(10, 10)
    y = torch.div(torch.ones(10, 10), torch.zeros(10, 10))
    harness(setting, Model, [x, y], trace_model)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("setting", {"default", "true", "false"})
@pytest.mark.parametrize("trace_model", [True, False])
def test_nonreal(setting, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.sqrt(x)

    x = torch.Tensor([-1, -2])
    harness(setting, Model, [x], trace_model)


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_nan(setting, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x > y

    x = torch.ones(10, 10)
    y = torch.div(torch.zeros(10, 10), torch.zeros(10, 10))
    harness(setting, Model, [x, y], trace_model)


@pytest.mark.parametrize("setting", {"default", "true", "false"})
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_ovf(setting, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.exp(x)

    x = torch.Tensor([3800, 4203])
    harness(setting, Model, [x], trace_model)
