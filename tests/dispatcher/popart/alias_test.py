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

    opts = poptorch.Options()
    opts.Jit.trace_model = False
    poptorch_model = poptorch.trainingModel(model, opts)

    x = torch.randn((2, ))
    poptorch_model.compile(x)

    helpers.assert_allequal(actual=model.layer_one.weight,
                            expected=model.layer_two.weight)
    helpers.assert_allequal(actual=model.layer_one.bias,
                            expected=model.layer_two.bias)
