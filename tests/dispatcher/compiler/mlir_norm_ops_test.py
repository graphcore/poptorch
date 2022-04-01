#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import copy
import torch
from torch import nn
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
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

    weights = {
        **dict(ipu_norm.named_parameters()),
        **dict(ipu_norm.named_buffers())
    }

    t2 = torch.ones_like(t)

    if affine:

        def cpu_step(norm, x1, x2):
            out = norm(x1)
            loss = torch.nn.functional.mse_loss(out, x2)
            loss.backward()
            return out, norm.weight.grad, norm.bias.grad
    else:
        cpu_step = lambda norm, x1, _: norm(x1)

    ipu_result = IPUContext(cpu_step, parameters_and_buffers=weights)(ipu_norm,
                                                                      t, t2)
    cpu_result = cpu_step(cpu_norm, t, t2)

    # Test outputs and gradients
    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(actual=ipu, expected=cpu)

    # Test running statistics
    if track_running_stats:
        helpers.assert_allclose(actual=ipu_norm.running_mean,
                                expected=cpu_norm.running_mean,
                                atol=1e-4,
                                rtol=1e-4)
        helpers.assert_allclose(actual=ipu_norm.running_var,
                                expected=cpu_norm.running_var,
                                atol=1e-4,
                                rtol=1e-4)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_group_norm():
    torch.manual_seed(42)
    num_groups = 4
    C = 12

    input_shape = [3, C, 5]
    norm = nn.GroupNorm(num_groups, C)

    t = torch.rand(input_shape)
    # Run pytorch native on CPU.
    torch_out = norm(t)

    weights = {**dict(norm.named_parameters()), **dict(norm.named_buffers())}
    # Run on IPU.
    ipu_result = IPUContext(norm, parameters_and_buffers=weights)(t)

    helpers.assert_allclose(actual=ipu_result, expected=torch_out)
