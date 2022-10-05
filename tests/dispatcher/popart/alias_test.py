#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch.nn as nn
import poptorch
import helpers


@pytest.mark.mlirSupportRequired
def test_alias():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_one = nn.Linear(2, 4)
            self.layer_two = nn.Linear(2, 4)

            self.layer_two.weight = self.layer_one.weight
            self.layer_two.bias = self.layer_one.bias

        def forward(self, x):
            x = self.layer_one(x)
            loss = poptorch.identity_loss(x, reduction="mean")
            return x, loss

    model = Model()

    poptorch_model = poptorch.trainingModel(model)

    x = torch.randn((2, ))
    poptorch_model.compile(x)

    helpers.assert_allequal(actual=model.layer_one.weight,
                            expected=model.layer_two.weight)
    helpers.assert_allequal(actual=model.layer_one.bias,
                            expected=model.layer_two.bias)


@pytest.mark.mlirSupportRequired
def test_alias_rewrap():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('y', torch.tensor([2.0]))
            self.layer_one = nn.Linear(2, 4)
            self.layer_two = nn.Linear(2, 4)

            self.layer_two.weight = self.layer_one.weight
            self.layer_two.bias = self.layer_one.bias

        def forward(self, x):
            # pylint doesn't understand register_buffer()
            # pylint: disable=no-member
            x = self.layer_one(x) * self.y
            self.y += 1.0
            loss = poptorch.identity_loss(x, reduction="mean")
            return x, loss

    model = Model()
    x = torch.randn((2, ))
    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.01)

    poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)
    poptorch_model(x)
    poptorch_model.destroy()

    # Create a new trainingModel which uses the same model and optimizer
    poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)
    poptorch_model(x)
