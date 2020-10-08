#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import poptorch
import torch


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
            x += 1
            x = poptorch.ipu_print_tensor(x) + self.bias
            return x, custom_loss(x, target)

    model = Model("MyAttr")
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([30.0, 40.0, 50.0])
    poptorch_model = poptorch.trainingModel(model)

    poptorch_model(input, target)

    assert poptorch_model.getAttr() == poptorch_model.attr
    assert poptorch_model.attr == "MyAttr"
