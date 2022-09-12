#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch


match_err_msg = \
    "The autocast API is not supported in PopTorch while using the dispatcher"


def test_autocast_decorator():
    class Model(torch.nn.Module):
        @poptorch.autocast()
        def forward(self, x, y):
            return torch.bmm(x, y)

    model = Model()

    torch.manual_seed(42)
    x = torch.randn(1, 20, 20)
    y = torch.randn(1, 20, 20)

    options = poptorch.Options()
    options.Jit.traceModel(False)
    poptorch_model = poptorch.inferenceModel(model)

    with pytest.raises(poptorch.Error, match=match_err_msg):
        poptorch_model(x, y)


def test_autocast_block():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            with poptorch.autocast():
                return torch.bmm(x, y)

    model = Model()

    torch.manual_seed(42)
    x = torch.randn(1, 20, 20)
    y = torch.randn(1, 20, 20)
    poptorch_model = poptorch.inferenceModel(model)

    with pytest.raises(poptorch.Error, match=match_err_msg):
        poptorch_model(x, y)


@pytest.mark.parametrize("setting", {"default", "true", "false"})
def test_enable_autocast(setting):
    torch.manual_seed(42)
    x = torch.randn(1, 1, 20, 20)
    model = torch.nn.Conv2d(1, 20, 5)
    model.autocast()

    opts = poptorch.Options()
    if setting == "true":
        opts.Precision.autocastEnabled(True)
    elif setting == "false":
        opts.Precision.autocastEnabled(False)

    poptorch_model = poptorch.inferenceModel(model)

    with pytest.raises(poptorch.Error, match=match_err_msg):
        poptorch_model(x)


@pytest.mark.parametrize("setting", {"hff", "default"})
def test_autocast_policy(setting):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 20, 20)

        def forward(self, x):
            return torch.relu(0.5 * self.conv.forward(x))

    torch.manual_seed(42)
    x = torch.randn(1, 1, 20, 20)
    model = Model()
    model.autocast()

    if setting == "hff":
        policy = poptorch.autocasting.Policy([torch.nn.Conv2d],
                                             [torch.mul, torch.relu])
    else:
        policy = poptorch.autocasting.Policy()

    opts = poptorch.Options()
    opts.Precision.autocastPolicy(policy)
    poptorch_model = poptorch.inferenceModel(model, opts)

    with pytest.raises(poptorch.Error, match=match_err_msg):
        poptorch_model(x)
