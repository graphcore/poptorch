#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
from torch import nn
import pytest
import helpers
import poptorch
from poptorch.enums import Compiler


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize(
    "params",
    [
        # Norm, affine, running_stats, training
        (nn.BatchNorm1d, False, False, False),
        (nn.BatchNorm2d, True, False, True),
        (nn.BatchNorm3d, False, True, True),
    ])
def test_batch_norm(params):
    torch.manual_seed(42)
    C = 4
    input_shape = [3, C, 5]
    batch_norm, affine, running_stats, training = params
    if batch_norm in (nn.BatchNorm2d, nn.BatchNorm3d):
        input_shape.append(6)
    if batch_norm is nn.BatchNorm3d:
        input_shape.append(7)
    t = torch.randn(input_shape)

    norm = batch_norm(C, affine=affine, track_running_stats=running_stats)

    # pylint: disable=W0212
    norm._buffers["running_mean"] = torch.randn([C])
    norm._buffers["running_var"] = torch.clamp(torch.randn([C]) + 1.0, min=0.1)
    norm.train(training)

    # Run pytorch native on CPU.
    torch_out = norm(t)

    weights = {**dict(norm.named_parameters()), **dict(norm.named_buffers())}

    # Run on IPU.
    with poptorch.IPUScope([t], weights, compile_using=Compiler.MLIR) as ipu:
        ipu.outputs([norm(t)])

    # pylint: disable=no-member
    helpers.assert_allclose(actual=ipu(t), expected=torch_out)
