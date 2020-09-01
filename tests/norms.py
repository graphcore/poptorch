#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.optim as optim
import poptorch
import pytest

# Norms
#'torch.nn.BatchNorm1d', 'torch.nn.BatchNorm2d', 'torch.nn.BatchNorm3d', 'torch.nn.GroupNorm', 'torch.nn.SyncBatchNorm', 'torch.nn.SyncBatchNorm.convert_sync_batchnorm',
# 'torch.nn.InstanceNorm1d', 'torch.nn.InstanceNorm2d', 'torch.nn.InstanceNorm3d', 'torch.nn.LayerNorm', 'torch.nn.LocalResponseNorm',

batch_norms = [
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d
]


@pytest.mark.parametrize("running_stats", {True, False})
@pytest.mark.parametrize("training", {True, False})
def test_batchNorm1D(running_stats, training):
    torch.manual_seed(42)

    input = torch.randn([2, 10, 1000])
    norm = torch.nn.BatchNorm1d(10, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([10])
    norm._buffers["running_var"] = torch.clamp(torch.randn([10]) + 1.0,
                                               min=0.1)
    norm.train(training)

    # Run pytorch native on CPU.
    nativeOutput = norm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(norm)
    poptorchOut = ipuModel(input)

    torch.testing.assert_allclose(poptorchOut,
                                  nativeOutput,
                                  atol=1e-1,
                                  rtol=0.1)


@pytest.mark.parametrize("running_stats", {True, False})
@pytest.mark.parametrize("training", {True, False})
def test_batchNorm2D(running_stats, training):
    torch.manual_seed(42)

    input = torch.randn([20, 10, 35, 45])
    norm = torch.nn.BatchNorm2d(10, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([10])
    norm._buffers["running_var"] = torch.clamp(torch.randn([10]) + 1.0,
                                               min=0.1)
    norm.train(training)

    # Run pytorch native on CPU.
    nativeOutput = norm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(norm)
    poptorchOut = ipuModel(input)

    torch.testing.assert_allclose(poptorchOut,
                                  nativeOutput,
                                  atol=1e-1,
                                  rtol=0.1)


@pytest.mark.parametrize("running_stats", {True, False})
@pytest.mark.parametrize("training", {True, False})
def test_batchNorm3D(running_stats, training):
    torch.manual_seed(42)

    input = torch.randn([20, 10, 35, 45, 10])
    norm = torch.nn.BatchNorm3d(10, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([10])
    norm._buffers["running_var"] = torch.clamp(torch.randn([10]) + 1.0,
                                               min=0.1)
    norm.train(training)

    # Run pytorch native on CPU.
    nativeOutput = norm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(norm)
    poptorchOut = ipuModel(input)

    torch.testing.assert_allclose(poptorchOut,
                                  nativeOutput,
                                  atol=1e-1,
                                  rtol=0.1)


def test_layerNorm():
    torch.manual_seed(42)

    for i in range(1, 4):
        input = torch.randn([3, 2, 5, 2])
        layerNorm = torch.nn.LayerNorm(input.size()[i:])

        # Run pytorch native on CPU.
        nativeOutput = layerNorm(input)

        # Run on IPU.
        ipuModel = poptorch.inferenceModel(layerNorm)
        poptorchOut = ipuModel(input)

        assert torch.allclose(poptorchOut, nativeOutput)


def test_layerNormScalar():
    torch.manual_seed(42)

    input = torch.randn([3, 2, 5, 2])
    layerNorm = torch.nn.LayerNorm(2)

    # Run pytorch native on CPU.
    nativeOutput = layerNorm(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(layerNorm)
    poptorchOut = ipuModel(input)

    assert torch.allclose(poptorchOut, nativeOutput)


def test_layerNormPretrainedWeights():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = torch.nn.Conv2d(5, 5, kernel_size=(1, 1))
            self.norm = torch.nn.LayerNorm((5, 3, 10))

        def forward(self, x):
            x = self.conv(x)

            return self.norm(x)

    model = Model()

    input = torch.randn([3, 5, 3, 10])

    modelOut = model(input)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(model)
    poptorchOut = ipuModel(input)

    # Marginally more leeway.
    assert torch.allclose(poptorchOut, modelOut, rtol=1e-4, atol=1e-6)

    # We aren't training to any real target we just want to update the beta/gamma parameters and check they still work in popart.
    criterion = torch.nn.MSELoss()
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
    poptorchOut = ipuModel(input)

    # Run on CPU again with trained weights.
    outputs = model(input)

    assert torch.allclose(poptorchOut, outputs, rtol=1e-4, atol=1e-6)


def test_groupNorm():
    torch.manual_seed(42)

    for _ in range(1, 4):
        input = torch.randn([3, 10, 5, 2])

        groupNorm = torch.nn.GroupNorm(5, 10)

        # Run pytorch native on CPU.
        nativeOutput = groupNorm(input)

        # Run on IPU.
        ipuModel = poptorch.inferenceModel(groupNorm)
        poptorchOut = ipuModel(input)

        # Group norm is pending correctness changes in popart/poplar so we will just test the shape/type for now.
        assert poptorchOut.size() == nativeOutput.size()
        assert poptorchOut.type() == nativeOutput.type()
