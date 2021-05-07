#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os  # pylint: disable=unused-import
import unittest.mock
import torch
import torch.optim as optim
import torch.nn as nn
import pytest
import helpers
import poptorch

# Norms
#'torch.nn.BatchNorm1d', 'torch.nn.BatchNorm2d', 'torch.nn.BatchNorm3d', 'torch.nn.GroupNorm', 'torch.nn.SyncBatchNorm', 'torch.nn.SyncBatchNorm.convert_sync_batchnorm',
# 'torch.nn.InstanceNorm1d', 'torch.nn.InstanceNorm2d', 'torch.nn.InstanceNorm3d', 'torch.nn.LayerNorm', 'torch.nn.LocalResponseNorm',

batch_norms = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]


@pytest.mark.parametrize("running_stats", {True, False})
@pytest.mark.parametrize("training", {True, False})
def test_batchNorm1D(running_stats, training):
    torch.manual_seed(42)

    input = torch.randn([2, 10, 1000])
    norm = nn.BatchNorm1d(10, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([10])
    norm._buffers["running_var"] = torch.clamp(torch.randn([10]) + 1.0,
                                               min=0.1)
    norm.train(training)

    # Run pytorch native on CPU.
    native_output = norm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(norm)
    poptorch_out = ipuModel(input)

    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_output,
                            atol=1e-1,
                            rtol=0.1)


@pytest.mark.parametrize("running_stats", {True, False})
@pytest.mark.parametrize("training", {True, False})
@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
def test_batchNorm2D(running_stats, training):
    torch.manual_seed(42)

    input = torch.randn([20, 10, 35, 45])
    norm = nn.BatchNorm2d(10, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([10])
    norm._buffers["running_var"] = torch.clamp(torch.randn([10]) + 1.0,
                                               min=0.1)
    norm.train(training)

    # Run pytorch native on CPU.
    native_output = norm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(norm)
    poptorch_out = ipuModel(input)

    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_output,
                            atol=1e-1,
                            rtol=0.1)


@pytest.mark.parametrize("running_stats", {True, False})
@pytest.mark.parametrize("training", {True, False})
@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
def test_batchNorm3D(running_stats, training):
    torch.manual_seed(42)

    input = torch.randn([20, 10, 35, 45, 10])
    norm = nn.BatchNorm3d(10, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([10])
    norm._buffers["running_var"] = torch.clamp(torch.randn([10]) + 1.0,
                                               min=0.1)
    norm.train(training)

    # Run pytorch native on CPU.
    native_output = norm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(norm)
    poptorch_out = ipuModel(input)

    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_output,
                            atol=1e-1,
                            rtol=0.1)


def test_batchNorm_eval_during_training():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.bn = nn.BatchNorm1d(100)
            self.loss = torch.nn.MSELoss()

        def forward(self, x, target):
            y = self.bn(x)
            return y, self.loss(y, target)

    input = torch.randn([16, 100])
    target = torch.randn([16, 100])

    model = Model()
    for param in model.parameters():
        param.requires_grad = False
    model.bn.eval()

    running_mean_init = model.bn.running_mean.clone().detach()
    running_var_init = model.bn.running_var.clone().detach()

    # Run pytorch native on CPU.
    native_out, _ = model(input, target)
    # Run on IPU.
    ipu_model = poptorch.trainingModel(model)
    poptorch_out, _ = ipu_model(input, target)
    # TODO: T38684
    # Implicit copy only happens when we touch the params so copy explicitly.
    ipu_model.copyWeightsToHost()

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
    helpers.assert_allequal(actual=model.bn.running_mean,
                            expected=running_mean_init)
    helpers.assert_allequal(actual=model.bn.running_var,
                            expected=running_var_init)


def test_layerNorm():
    torch.manual_seed(42)

    for i in range(1, 4):
        input = torch.randn([3, 2, 5, 2])
        layerNorm = nn.LayerNorm(input.size()[i:])

        # Run pytorch native on CPU.
        native_output = layerNorm(input)

        # Run on IPU.
        ipuModel = poptorch.inferenceModel(layerNorm)
        poptorch_out = ipuModel(input)

        helpers.assert_allclose(actual=poptorch_out, expected=native_output)


def test_layerNormScalar():
    torch.manual_seed(42)

    input = torch.randn([3, 2, 5, 2])
    layerNorm = nn.LayerNorm(2)

    # Run pytorch native on CPU.
    native_output = layerNorm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(layerNorm)
    poptorch_out = ipuModel(input)

    helpers.assert_allclose(actual=poptorch_out, expected=native_output)


def test_layerNormPretrainedWeights():
    torch.manual_seed(42)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(5, 5, kernel_size=(1, 1))
            self.norm = nn.LayerNorm((5, 3, 10))

        def forward(self, x):
            x = self.conv(x)

            return self.norm(x)

    model = Model()

    input = torch.randn([3, 5, 3, 10])

    modelOut = model(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(model)
    poptorch_out = ipuModel(input)

    # Marginally more leeway.
    helpers.assert_allclose(actual=poptorch_out,
                            expected=modelOut,
                            rtol=1e-4,
                            atol=1e-6)

    # We aren't training to any real target we just want to update the beta/gamma parameters and check they still work in popart.
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for _ in range(0, 10):
        outputs = model(input)
        optimizer.zero_grad()
        loss = criterion(outputs, torch.ones([3, 5, 3, 10]))
        loss.backward()
        optimizer.step()

    model.eval()
    # Run on IPU with trained weights.
    ipuModel = poptorch.inferenceModel(model)
    poptorch_out = ipuModel(input)

    # Run on CPU again with trained weights.
    outputs = model(input)

    helpers.assert_allclose(actual=poptorch_out,
                            expected=outputs,
                            rtol=1e-4,
                            atol=1e-6)


@pytest.mark.parametrize("dims", {2, 3, 4, 5})
def test_groupNorm(dims):
    torch.manual_seed(42)

    shape = [3, 10]
    if dims > 2:
        rand_shape = torch.randint(2, 5, [dims - 2])
        shape.extend(rand_shape.tolist())

    for _ in range(3):
        input = torch.randn(shape)
        groupNorm = nn.GroupNorm(5, 10)

        # Run pytorch native on CPU.
        native_output = groupNorm(input)

        # Run on IPU.
        ipuModel = poptorch.inferenceModel(groupNorm)
        poptorch_out = ipuModel(input)

        # Group norm is pending correctness changes in popart/poplar so we will just test the shape/type for now.
        assert poptorch_out.size() == native_output.size()
        assert poptorch_out.type() == native_output.type()


@pytest.mark.parametrize("instanceNormXd", {(nn.InstanceNorm1d, 1),
                                            (nn.InstanceNorm2d, 2),
                                            (nn.InstanceNorm3d, 3)})
def test_instanceNorm(instanceNormXd):
    torch.manual_seed(42)

    d = instanceNormXd[1]

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.norm = instanceNormXd[0](6, affine=True)
            self.fc1 = nn.Linear(6 * 2**d, 10)

        def forward(self, x):
            x = self.norm(x)
            x = x.flatten(1)
            return self.fc1(x)

    for _ in range(3):
        model = Model()
        opt = optim.AdamW(model.parameters(), lr=0.01)

        poptorch_model = helpers.trainingModelWithLoss(
            model, loss=nn.CrossEntropyLoss(), optimizer=opt)

        shape = [5, 6]
        shape.extend([2 for i in range(d)])

        # Offset the data by multiplying by random values and shifting by a random bias
        input = torch.randint(2, 10, shape) * torch.randn(
            shape) + torch.randint(2, 10, [1]) * torch.randn(1)
        label = torch.randint(0, 10, [shape[0]])

        _, original_loss = poptorch_model(input, label)

        for _ in range(0, 100):
            out, loss = poptorch_model(input, label)

        # Check we have trained the model
        assert loss < original_loss
        assert loss < 0.03
        helpers.assert_allequal(actual=torch.argmax(out, dim=1),
                                expected=label)


def test_batchnorm_statistics():
    torch.manual_seed(42)

    input_data = [torch.randn([4, 4, 3, 3]) for _ in range(10)]
    label = torch.ones(4).long()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(4)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, args, loss_inputs=None):
            output = self.bn(args)
            if loss_inputs is None:
                return output

            reduced = torch.mean(output, dim=(2, 3))
            return output, self.loss(reduced, loss_inputs)

    model1 = Model()
    model1.train()
    optimizer = optim.SGD(model1.parameters(), lr=0.0)
    model_opts = poptorch.Options()
    training_model = poptorch.trainingModel(model1,
                                            model_opts,
                                            optimizer=optimizer)

    for data in input_data:
        training_model(data, label)

    model2 = Model()
    model2.train()
    for data in input_data:
        model2(data)

    # Shouldn't be needed but buffers alone don't trigger the copy.
    training_model.copyWeightsToHost()

    # Running mean is very close
    helpers.assert_allclose(actual=model2.bn.running_mean,
                            expected=model1.bn.running_mean)

    # Running var is not so close.
    helpers.assert_allclose(actual=model2.bn.running_var,
                            expected=model1.bn.running_var,
                            atol=1e-1,
                            rtol=0.1)
