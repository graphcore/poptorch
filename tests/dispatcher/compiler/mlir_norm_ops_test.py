#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import torch
from torch import nn
import pytest
import helpers
from poptorch.experimental import IPUContext


@pytest.mark.parametrize("batch_norm, affine, track_running_stats, training", [
    (nn.BatchNorm1d, True, False, False),
    (nn.BatchNorm2d, True, False, True),
    (nn.BatchNorm3d, True, True, True),
    (nn.BatchNorm1d, False, False, False),
    (nn.BatchNorm2d, False, False, True),
    (nn.BatchNorm3d, False, True, True),
])
def test_batch_norm(batch_norm, affine, track_running_stats, training):
    torch.manual_seed(42)
    C = 4
    input_shape = [3, C, 5]
    if batch_norm in (nn.BatchNorm2d, nn.BatchNorm3d):
        input_shape.append(6)
    if batch_norm is nn.BatchNorm3d:
        input_shape.append(7)
    t = torch.randn(input_shape)

    ipu_norm = batch_norm(C,
                          affine=affine,
                          track_running_stats=track_running_stats)
    ipu_norm.train(training)
    cpu_norm = copy.deepcopy(ipu_norm)

    t2 = torch.ones_like(t)

    def cpu_step(norm, x1, x2):
        x1.requires_grad = True
        x1.retain_grad()
        out = norm(x1)
        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()
        ret = [out, x1.grad]
        if affine:
            ret.extend((norm.weight.grad, norm.bias.grad))
        if track_running_stats:
            ret.append(norm.running_mean)
        return ret

    ipu_result = IPUContext(cpu_step, model=ipu_norm)(ipu_norm, t, t2)
    cpu_result = cpu_step(cpu_norm, t, t2)

    # Test outputs and gradients
    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(actual=ipu, expected=cpu)


@pytest.mark.parametrize("num_dims", (2, 3, 4))
@pytest.mark.parametrize("affine", (True, False))
def test_group_norm(num_dims, affine):
    torch.manual_seed(42)

    num_groups = 4
    C = 12

    if num_dims == 2:
        input_shape = [3, C]
    if num_dims == 3:
        input_shape = [3, C, 5]
    if num_dims == 4:
        input_shape = [3, C, 5, 4]

    ipu_norm = nn.GroupNorm(num_groups, C)

    # Ensure the params are not default (0 and 1)
    if affine:
        torch.nn.init.uniform_(ipu_norm.weight, 0.1, 2.0)
        torch.nn.init.uniform_(ipu_norm.bias, -1.0, 1.0)

    cpu_norm = copy.deepcopy(ipu_norm)

    t = torch.rand(input_shape)
    # Make the mean and stdev varied throughout
    t = t + torch.randint(0, 5, input_shape)

    t2 = torch.ones_like(t)

    def cpu_step(norm, x1, x2):
        x1.requires_grad = True
        x1.retain_grad()
        out = norm(x1)
        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()
        ret = [out, x1.grad, norm.weight.grad, norm.bias.grad]
        return ret

    ipu_result = IPUContext(cpu_step, model=ipu_norm)(ipu_norm, t, t2)
    cpu_result = cpu_step(cpu_norm, t, t2)

    # Test outputs and gradients
    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(actual=ipu, expected=cpu, rtol=1e-3, atol=1e-3)


def test_group_norm_train():
    torch.manual_seed(42)
    kernel_size = 3
    num_groups = 4
    C = 12

    input_shape = [3, C, 5]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(C, C, kernel_size)
            self.norm = nn.GroupNorm(num_groups, C)

        def forward(self, x, target):
            out = self.conv(x)
            out = self.norm(out)
            loss = torch.nn.functional.mse_loss(out, target)

            return out, loss

    model = Model()

    torch.nn.init.uniform_(model.norm.weight, 0.1, 2.0)
    torch.nn.init.uniform_(model.norm.bias, -1.0, 1.0)

    cpu_model = copy.deepcopy(model)

    t = torch.rand(input_shape)

    # Make the mean and stdev varied throughout
    t = t + torch.randint(0, 5, input_shape)

    target = torch.rand(
        [input_shape[0], input_shape[1], input_shape[2] - kernel_size + 1])

    def grad(model, t, target):
        _, loss = model(t, target)
        loss.backward()

        model.conv.weight.retain_grad()
        model.conv.bias.retain_grad()

        return (loss, model.conv.weight.grad, model.conv.bias.grad,
                model.norm.weight.grad, model.norm.bias.grad)

    # Run on IPU.
    ipu_result = IPUContext(grad, model=model)(model, t, target)

    # Run pytorch native on CPU.
    cpu_out = grad(cpu_model, t, target)

    assert len(ipu_result) == len(cpu_out)

    for idx, cpu_out_exp in enumerate(cpu_out):
        helpers.assert_allclose(actual=ipu_result[idx], expected=cpu_out_exp)


@pytest.mark.parametrize(
    "instance_norm, affine, track_running_stats, training", [
        (nn.InstanceNorm1d, True, False, False),
        (nn.InstanceNorm2d, True, False, True),
        (nn.InstanceNorm3d, True, True, True),
        (nn.InstanceNorm1d, False, False, False),
        (nn.InstanceNorm2d, False, False, True),
        (nn.InstanceNorm3d, False, True, True),
    ])
def test_instance_norm(instance_norm, affine, track_running_stats, training):
    # (Instance norm is cast to batch_norm within PyTorch)

    torch.manual_seed(42)
    C = 4
    input_shape = [3, C, 5]
    if instance_norm in (nn.InstanceNorm2d, nn.InstanceNorm3d):
        input_shape.append(6)
    if instance_norm is nn.InstanceNorm3d:
        input_shape.append(7)

    t = torch.rand(input_shape)
    # Make the mean and stdev varied throughout
    t = t + torch.randint(0, 5, input_shape)

    ipu_norm = instance_norm(C,
                             affine=affine,
                             track_running_stats=track_running_stats)

    # Ensure the params are not default (0 and 1)
    if affine:
        torch.nn.init.uniform_(ipu_norm.weight, 0.1, 2.0)
        torch.nn.init.uniform_(ipu_norm.bias, -1.0, 1.0)

    ipu_norm.train(training)
    cpu_norm = copy.deepcopy(ipu_norm)

    t2 = torch.ones_like(t)

    def cpu_step(norm, x1, x2):
        x1.requires_grad = True
        print("Before retain_grad")
        x1.retain_grad()
        print("After retain_grad")

        out = norm(x1)
        loss = torch.nn.functional.mse_loss(out, x2)
        print("Before backward")
        loss.backward()
        print("after backward")
        ret = [out, x1.grad]
        print("After set ret")
        if affine:
            ret.extend((norm.weight.grad, norm.bias.grad))
        if track_running_stats:
            ret.append(norm.running_mean)
        print("Before return")
        return ret

    ipu_result = IPUContext(cpu_step, model=ipu_norm)(ipu_norm, t, t2)
    cpu_result = cpu_step(cpu_norm, t, t2)

    # Test outputs and gradients
    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(actual=ipu, expected=cpu)


@pytest.mark.parametrize("num_dims_normalize_start", [(2, 0), (2, 1), (3, 0),
                                                      (3, 1), (3, 2), (4, 0),
                                                      (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("affine", (True, False))
def test_layer_norm(num_dims_normalize_start, affine):
    torch.manual_seed(42)
    (num_dims, normalize_start) = num_dims_normalize_start

    if num_dims == 2:
        input_shape = [3, 4]
    if num_dims == 3:
        input_shape = [3, 4, 5]
    if num_dims == 4:
        input_shape = [3, 4, 5, 4]

    normalized_shape = input_shape[normalize_start:]

    ipu_norm = nn.LayerNorm(normalized_shape, elementwise_affine=affine)

    # Ensure the params are not default (0 and 1)
    if affine:
        torch.nn.init.uniform_(ipu_norm.weight, 0.1, 2.0)
        torch.nn.init.uniform_(ipu_norm.bias, -1.0, 1.0)

    cpu_norm = copy.deepcopy(ipu_norm)

    t = torch.rand(input_shape)
    # Make the mean and stdev varied throughout
    t = t + torch.randint(0, 5, input_shape)

    t2 = torch.ones_like(t)

    def cpu_step(norm, x1, x2):
        x1.requires_grad = True
        x1.retain_grad()
        out = norm(x1)
        loss = torch.nn.functional.mse_loss(out, x2)
        loss.backward()
        ret = [out, x1.grad]
        if affine:
            ret.extend((norm.weight.grad, norm.bias.grad))
        return ret

    ipu_result = IPUContext(cpu_step, model=ipu_norm)(ipu_norm, t, t2)
    cpu_result = cpu_step(cpu_norm, t, t2)

    # Test outputs and gradients
    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(actual=ipu, expected=cpu, rtol=1e-3, atol=1e-3)
