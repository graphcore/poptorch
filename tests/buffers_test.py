#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import helpers
import poptorch


class ConstantBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('stuff', torch.tensor([1, 2, 3],
                                                   dtype=torch.int32))

    def forward(self, x):
        new_stuff = 1.0 + self.stuff
        return torch.sum(x + new_stuff)


def test_constant_buffer():
    model = ConstantBuffer()

    poptorch_model = poptorch.inferenceModel(model)
    assert poptorch_model(torch.tensor([2])) == 15


def test_constant_buffer_repeat():
    model = ConstantBuffer()

    poptorch_model = poptorch.inferenceModel(model)
    assert poptorch_model(torch.tensor([2])) == 15
    assert poptorch_model(torch.tensor([2])) == 15


def test_buffer_implicit_copy():
    momentum = 0.1

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.bn = torch.nn.BatchNorm1d(10, momentum=momentum)
            self.loss = torch.nn.MSELoss()

        def forward(self, x, target):
            y = self.bn(x)
            return y, self.loss(y, target)

    model = Model()

    input = torch.ones([4, 10], dtype=torch.float32)
    target = torch.ones([4, 10], dtype=torch.float32) + 1

    poptorch_model = poptorch.trainingModel(model)

    poptorch_model(input, target)
    helpers.assert_allclose(actual=model.bn.running_mean,
                            expected=input[0, :] * momentum)

    poptorch_model.copyWeightsToHost()
    helpers.assert_allclose(actual=model.bn.running_mean,
                            expected=input[0, :] * momentum)
