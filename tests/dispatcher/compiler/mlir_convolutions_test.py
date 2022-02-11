#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext

to_test = [
    # input_dims, out_channels, kernel_size, stride, padding, dilation
    {
        "input": (1, 3, 20, 10, 10),
        "out_channels": 5,
        "kernel": (2, 4, 5)
    },
    {
        "input": (1, 3, 20, 10, 10),
        "out_channels": 5,
        "kernel": (2, 4, 5),
        "bias": False
    },
    {
        "input": (1, 2, 15, 12, 12),
        "out_channels": 2,
        "kernel": (3, 6, 6),
        "strides": (2, 2, 2)
    },
    {
        "input": (1, 2, 15, 12, 25),
        "out_channels": 2,
        "kernel": (10, 2, 5),
        "strides": (1, 3, 4),
        "padding": (1, 4, 4)
    },
    {
        "input": (1, 2, 15, 18, 20),
        "out_channels": 2,
        "kernel": (1, 2, 3),
        "strides": (1, 3, 2),
        "padding": (3, 4, 3),
        "dilation": (2, 3, 1)
    },
    {
        "input": (1, 2, 30, 23, 10),
        "out_channels": 2,
        "kernel": (1, 2, 4),
        "strides": (1, 3, 2),
        "padding": (1, 4, 3),
        "dilation": (4, 1, 3),
        "groups": 2
    },
]


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("size", to_test)
def test_conv(num_dims, size):
    torch.manual_seed(42)

    input_dim = size["input"]
    out_channels = size["out_channels"]
    kernel_size = size["kernel"]

    strides = size.get("strides", (1, ) * 3)
    padding = size.get("padding", (0, ) * 3)
    dilation = size.get("dilation", (1, ) * 3)
    groups = size.get("groups", 1)
    bias = size.get("bias", True)

    if num_dims != 3:
        kernel_size = kernel_size[:num_dims - 3]
        input_dim = input_dim[:num_dims - 3]
        strides = strides[:num_dims - 3]
        padding = padding[:num_dims - 3]
        dilation = dilation[:num_dims - 3]

    t1 = torch.randn(input_dim)
    in_channels = t1.size()[1]

    if num_dims == 1:
        conv = torch.nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size,
                               strides,
                               padding,
                               dilation,
                               groups,
                               bias=bias)
    elif num_dims == 2:
        conv = torch.nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               strides,
                               padding,
                               dilation,
                               groups,
                               bias=bias)
    elif num_dims == 3:
        conv = torch.nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size,
                               strides,
                               padding,
                               dilation,
                               groups,
                               bias=bias)

    ipu_result = IPUContext(conv,
                            parameters_and_buffers=conv.named_parameters())(t1)
    cpu_result = conv(t1)

    assert ipu_result.size() == cpu_result.size()

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result,
                            actual=cpu_result,
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)
