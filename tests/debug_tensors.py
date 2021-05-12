#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import torch
import poptorch
import helpers


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
        self.loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, x, labels=None):
        out = self.fc2(self.relu(self.fc1(x)))
        if self.training:
            return self.loss(out, labels)
        return out


def test_tensor_names():
    model = Model()
    poptorch_model = poptorch.trainingModel(model)
    input = torch.rand(10, 10)
    label = torch.rand(10, 10)

    with pytest.raises(AssertionError):
        poptorch_model.getTensorNames()

    poptorch_model(input, label)
    tensors = poptorch_model.getTensorNames()

    assert any([t.startswith('Gradient___') for t in tensors])
    assert any([t.startswith('UpdatedVar__') for t in tensors])
    assert any([t.startswith('scaledLearningRate') for t in tensors])


def test_tensor_values():
    model = Model()

    opts = poptorch.Options()
    opts.anchorTensor('grad_bias', 'Gradient___model.fc2.bias')
    opts.anchorTensor('update_weight', 'UpdatedVar___model.fc2.weight')
    poptorch_model = poptorch.trainingModel(model, opts)

    input = torch.rand(10, 10)
    label = torch.rand(10, 10)
    poptorch_model(input, label)

    grad1 = poptorch_model.getAnchoredTensor('grad_bias')
    assert grad1.shape == (10, )
    update1 = poptorch_model.getAnchoredTensor('update_weight')
    assert update1.shape == (10, 10)

    input = torch.rand(10, 10)
    label = torch.rand(10, 10)
    poptorch_model(input, label)

    grad2 = poptorch_model.getAnchoredTensor('grad_bias')
    assert grad2.shape == (10, )
    update2 = poptorch_model.getAnchoredTensor('update_weight')
    assert update2.shape == (10, 10)

    assert not torch.equal(grad1, grad2)
    assert not torch.equal(update1, update2)


anchor_modes = [[poptorch.AnchorMode.All, 3, "ALL/1"],
                [poptorch.AnchorMode.EveryN, 4, "EVERYN/4"],
                [poptorch.AnchorMode.Final, 1, "FINAL/1"],
                [poptorch.AnchorMode.Sum, 1, "Sum/1"]]


@pytest.mark.parametrize("mode", anchor_modes)
@helpers.printCapfdOnExit
def test_tensor_modes(capfd, mode):
    model = Model()
    tensor_name = 'Gradient___model.fc2.bias'

    poptorch.setLogLevel("DEBUG")
    opts = poptorch.Options()
    opts.anchorTensor('grad_bias', tensor_name, mode[0], mode[1])
    poptorch_model = poptorch.trainingModel(model, opts)

    input = torch.rand(10, 10)
    label = torch.rand(10, 10)
    poptorch_model(input, label)

    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains(tensor_name + ' ' + mode[2])
