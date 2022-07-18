#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import re
import tempfile
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


@pytest.mark.parametrize("trace_model", [True, False])
def test_tensor_names(trace_model):
    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)
    input = torch.rand(10, 10)
    label = torch.rand(10, 10)

    with pytest.raises(AssertionError):
        poptorch_model.getTensorNames()

    poptorch_model(input, label)
    tensors = poptorch_model.getTensorNames()

    assert any([re.search(r"\bfc1\b", t) for t in tensors])
    assert any([re.search(r"\bfc2\b", t) for t in tensors])
    assert any([t.startswith('input') for t in tensors])
    assert any([t.startswith('loss') for t in tensors])
    assert any([t.startswith('Gradient___') for t in tensors])
    assert any([t.startswith('UpdatedVar__') for t in tensors])
    assert any([t.startswith('scaledLearningRate') for t in tensors])
    assert any([t.startswith('weightDecayScaleFactor') for t in tensors])


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_tensor_names_from_precompiled_model(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        model = Model()
        options = poptorch.Options()
        options.Jit.traceModel(trace_model)
        poptorch_model = poptorch.trainingModel(model, options=options)
        input = torch.rand(10, 10)
        label = torch.rand(10, 10)

        # Running the model will trigger the executable compilation
        poptorch_model(input, label)
        # Save the executable and destroy the model
        poptorch_model.save(filename)
        poptorch_model.destroy()

        with pytest.raises(AssertionError):
            poptorch_model.getTensorNames()

        # Reload the model from file.
        poptorch_model = poptorch.load(filename)

        tensors = poptorch_model.getTensorNames()

        assert any([re.search(r"\bfc1\b", t) for t in tensors])
        assert any([re.search(r"\bfc2\b", t) for t in tensors])
        assert any([t.startswith('input') for t in tensors])
        assert any([t.startswith('loss') for t in tensors])
        assert any([t.startswith('weightDecayScaleFactor') for t in tensors])
        assert any([t.startswith('scaledLearningRate') for t in tensors])


@pytest.mark.parametrize("trace_model", [True, False])
def test_tensor_values(trace_model):
    model = Model()

    # The optimizer wrapper in the training model add an extra prefix to the
    # parameter names
    model_prefix = 'model.' if trace_model else ''

    opts = poptorch.Options()
    opts.anchorTensor('grad_bias', f'Gradient___{model_prefix}fc2.bias')
    opts.anchorTensor('update_weight',
                      f'UpdatedVar___{model_prefix}fc2.weight')
    opts.Jit.traceModel(trace_model)
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


output_modes = [[poptorch.OutputMode.All, 3, "ALL/1"],
                [poptorch.OutputMode.EveryN, 4, "EVERYN/4"],
                [poptorch.OutputMode.Final, 1, "FINAL/1"],
                [poptorch.OutputMode.Sum, 1, "Sum/1"]]


@pytest.mark.parametrize("mode, period, expected_str", output_modes)
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_tensor_modes(capfd, mode, period, expected_str, trace_model):
    # The optimizer wrapper in the training model add an extra prefix to the
    # parameter names
    model_prefix = 'model.' if trace_model else ''

    model = Model()
    tensor_name = f'Gradient___{model_prefix}fc2.bias'

    opts = poptorch.Options()
    opts.anchorTensor('grad_bias', tensor_name, mode, period)
    opts.Jit.traceModel(trace_model)

    poptorch_model = poptorch.trainingModel(model, opts)

    input = torch.rand(10, 10)
    label = torch.rand(10, 10)
    poptorch_model(input, label)

    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains(tensor_name + ' ' + expected_str)
