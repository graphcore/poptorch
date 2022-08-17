#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os  # pylint: disable=unused-import
import unittest.mock
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn as nn
import pytest
import helpers
import poptorch

# Norms
#'torch.nn.BatchNorm1d', 'torch.nn.BatchNorm2d', 'torch.nn.BatchNorm3d', 'torch.nn.GroupNorm', 'torch.nn.SyncBatchNorm', 'torch.nn.SyncBatchNorm.convert_sync_batchnorm',
# 'torch.nn.InstanceNorm1d', 'torch.nn.InstanceNorm2d', 'torch.nn.InstanceNorm3d', 'torch.nn.LayerNorm', 'torch.nn.LocalResponseNorm',

batch_norm_params = [
    # Norm, affine, running_stats, training, trace_model
    (nn.BatchNorm1d, False, False, False, True),
    (nn.BatchNorm1d, False, False, False, False),
    (nn.BatchNorm2d, True, True, True, True),
    (nn.BatchNorm2d, True, True, False, False),
    (nn.BatchNorm3d, False, True, True, True),
]


@pytest.mark.parametrize(
    "batch_norm, affine, running_stats, training, trace_model",
    batch_norm_params)
@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
def test_batchNorm(batch_norm, affine, running_stats, training, trace_model):
    torch.manual_seed(42)
    C = 4
    input_shape = [3, C, 5]
    if batch_norm in (nn.BatchNorm2d, nn.BatchNorm3d):
        input_shape.append(6)
    if batch_norm is nn.BatchNorm3d:
        input_shape.append(7)
    input = torch.randn(input_shape)

    norm = batch_norm(C, affine=affine, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([C])
    norm._buffers["running_var"] = torch.clamp(torch.randn([C]) + 1.0, min=0.1)
    norm.train(training)

    model = helpers.ModelWithWeights(norm, input.shape)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipumodel = deepcopy(model)
    poptorch_model = poptorch.trainingModel(
        ipumodel, options) if training else poptorch.inferenceModel(
            ipumodel, options)

    # Run pytorch native on CPU.
    native_out, _ = model((input, ))

    # Run on IPU.
    poptorch_out, _ = poptorch_model((input, ))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    # Training test - check weights changed
    if training:
        poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("trace_model", [True, False])
def test_batchNorm_eval_during_training(trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipu_model = poptorch.trainingModel(model, options=options)
    poptorch_out, _ = ipu_model(input, target)
    # TODO: T38684
    # Implicit copy only happens when we touch the params so copy explicitly.
    ipu_model.copyWeightsToHost()

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
    helpers.assert_allequal(actual=model.bn.running_mean,
                            expected=running_mean_init)
    helpers.assert_allequal(actual=model.bn.running_var,
                            expected=running_var_init)


@pytest.mark.parametrize("norm_dim", range(4))
@pytest.mark.parametrize("trace_model", [True, False])
def test_layerNorm(norm_dim, trace_model):
    torch.manual_seed(42)

    elementwise_affine = norm_dim % 2 == 1

    input = torch.randn([3, 2, 5, 2])
    layerNorm = nn.LayerNorm(input.shape[norm_dim:],
                             elementwise_affine=elementwise_affine)

    model = helpers.ModelWithWeights(layerNorm, input.shape)

    # Run pytorch native on CPU.
    native_out, _ = model((input, ))

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)
    # Run on IPU.
    poptorch_out, _ = poptorch_model((input, ))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out,
                            expected=native_out,
                            atol=1e-4,
                            rtol=1e-4)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("trace_model", [True, False])
def test_layerNormPretrainedWeights(trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuModel = poptorch.inferenceModel(model, options)
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
    ipuModel = poptorch.inferenceModel(model, options)
    poptorch_out = ipuModel(input)

    # Run on CPU again with trained weights.
    outputs = model(input)

    helpers.assert_allclose(actual=poptorch_out,
                            expected=outputs,
                            rtol=1e-4,
                            atol=1e-6)


@pytest.mark.parametrize("dims", {2, 3, 4, 5})
@pytest.mark.parametrize("trace_model", [True, False])
def test_groupNorm(dims, trace_model):
    if dims == 2:
        # TODO(T49073): Match torch 1.10 GroupNorm implementation
        pytest.skip("Numerical differences between PyTorch and PopTorch")

    torch.manual_seed(42)

    affine = dims % 2 == 0

    shape = [3, 10]
    if dims > 2:
        rand_shape = torch.randint(2, 5, [dims - 2])
        shape.extend(rand_shape.tolist())

    input = torch.randn(shape)
    groupNorm = nn.GroupNorm(5, 10, affine=affine)
    model = helpers.ModelWithWeights(groupNorm, input.shape)

    # Run pytorch native on CPU.
    native_out, _ = model((input, ))

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)
    poptorch_out, _ = poptorch_model((input, ))

    # Inference test - check outputs
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("trace_model", [True, False])
def test_groupNorm_exfail(trace_model):
    torch.manual_seed(42)

    shape = [3, 10]

    input = torch.randn(shape)
    groupNorm = nn.GroupNorm(5, 10)

    # Run pytorch native on CPU.
    native_output = groupNorm(input)

    opts = poptorch.Options()
    opts._Popart.set("groupNormStridedChannelGrouping", True)  # pylint: disable=protected-access
    opts.Jit.traceModel(trace_model)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(groupNorm, opts)
    poptorch_out = ipuModel(input)

    # Group norm is pending correctness changes in popart/poplar so we will just test the shape/type for now.
    assert poptorch_out.size() == native_output.size()
    assert poptorch_out.type() == native_output.type()

    assert not torch.allclose(poptorch_out, native_output, atol=1e-1, rtol=0.1)


instance_norm_params = [
    # norm, dims
    (nn.InstanceNorm1d, 1),
    (nn.InstanceNorm2d, 2),
    (nn.InstanceNorm3d, 3)
]


@pytest.mark.parametrize("instance_norm, d", instance_norm_params)
@pytest.mark.parametrize("trace_model", [True, False])
def test_instanceNorm(instance_norm, d, trace_model):
    torch.manual_seed(42)

    affine = d % 2 == 1

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.norm = instance_norm(6, affine=affine)
            self.fc1 = nn.Linear(6 * 2**d, 10)
            self.loss = nn.CrossEntropyLoss()

        def forward(self, x, target):
            out = self.norm(x)
            out = out.flatten(1)
            out = self.fc1(out)
            loss = self.loss(out, target)

            return out, loss

    for _ in range(3):
        model = Model()
        opt = optim.AdamW(model.parameters(), lr=0.01)
        options = poptorch.Options()
        options.Jit.traceModel(trace_model)
        poptorch_model = poptorch.trainingModel(model,
                                                options=options,
                                                optimizer=opt)

        shape = [5, 6]
        shape.extend([2 for _ in range(d)])

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


@pytest.mark.parametrize("trace_model", [True, False])
def test_batchnorm_statistics(trace_model):
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
    model_opts.Jit.traceModel(trace_model)
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
                            expected=model1.bn.running_var)
