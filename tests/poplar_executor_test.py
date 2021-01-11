#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import tempfile
import pytest
import torch
import helpers
import poptorch


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
@helpers.printCapfdOnExit
def test_ExecutableCaching(capfd):
    poptorch.setLogLevel(1)  # Force debug logging

    class Model(torch.nn.Module):
        def forward(self, x):
            return x * 6

    with tempfile.TemporaryDirectory() as cache:
        opts = poptorch.Options()
        opts.enableExecutableCaching(cache)
        m = poptorch.inferenceModel(Model(), opts)
        m.compile(torch.rand(2, 3))
        m.destroy()
        log = helpers.LogChecker(capfd)
        log.assert_contains("set enableEngineCaching to value true")
        assert os.listdir(), "No executable saved in the cache"

        n = poptorch.inferenceModel(Model(), opts)
        n.compile(torch.rand(2, 3))
        log = helpers.LogChecker(capfd)
        log.assert_contains("set enableEngineCaching to value true")


def test_inference_attributes():
    class Model(torch.nn.Module):
        def __init__(self, attr):
            super().__init__()
            self.attr = attr

        def getAttr(self):
            return self.attr

        def forward(
                self,
                x,
                y,
        ):
            return x + y + 5

    poptorch_model = poptorch.inferenceModel(Model("MyAttr"))

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    poptorch_model(t1, t2)

    assert poptorch_model.getAttr() == poptorch_model.attr
    assert poptorch_model.attr == "MyAttr"


def test_training_attributes():
    def custom_loss(output, target):
        # Mean squared error with a scale
        loss = output - target
        loss = loss * loss * 5
        return poptorch.identity_loss(loss, reduction="mean")

    class Model(torch.nn.Module):
        def __init__(self, attr):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))
            self.attr = attr

        def getAttr(self):
            return self.attr

        def forward(self, x, target):
            x = x + 1
            x = poptorch.ipu_print_tensor(x) + self.bias
            return x, custom_loss(x, target)

    model = Model("MyAttr")
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([30.0, 40.0, 50.0])
    poptorch_model = poptorch.trainingModel(model)

    poptorch_model(input, target)

    assert poptorch_model.getAttr() == poptorch_model.attr
    assert poptorch_model.attr == "MyAttr"


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="We need to be able to lock a specific IPU")
@pytest.mark.parametrize("use_half", [False])
def test_explicit_deletion(use_half):
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            x = x + 1

            # It is important to make sure the result of the print is used.
            x = poptorch.ipu_print_tensor(x)

            return x + self.bias

    def custom_loss(output, target):
        # Mean squared error with a scale
        loss = output - target
        loss = loss * loss * 5
        return poptorch.identity_loss(loss, reduction="mean")

    class ExampleModelWithCustomLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ExampleModel()

        def forward(self, input, target=None):
            out = self.model(input)
            if target is not None:
                return out, custom_loss(out, target)
            return out

    opts = poptorch.Options()
    # Both models will use the same IPU device.
    opts.useIpuId(1)

    model = ExampleModelWithCustomLoss()
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([30.0, 40.0, 50.0])
    if use_half:
        model.half()
        input = input.half()
        target = target.half()
    training_model = poptorch.trainingModel(model, opts)
    inference_model = poptorch.inferenceModel(model, opts)

    training_model(input=input, target=target)
    training_model.destroy()

    inference_model(input)
