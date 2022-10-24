#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn as nn
import pytest
import helpers
from poptorch.experimental import IPUContext


@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize(
    "input_size, out_channels, kernel_size, stride, padding, dilation, "
    "groups, bias", [
        ((1, 3, 20, 10, 10), 5, (2, 4, 5), None, None, None, None, None),
        ((1, 3, 20, 10, 10), 5, (2, 4, 5), None, None, None, None, False),
        ((1, 2, 15, 12, 12), 2, (3, 6, 6), (2, 2, 2), None, None, None, None),
        ((1, 2, 15, 12, 25), 2, (10, 2, 5), (1, 3, 4),
         (1, 4, 4), None, None, None),
        ((1, 2, 15, 18, 20), 2, (1, 2, 3), (1, 3, 2), (3, 4, 3),
         (2, 3, 1), None, None),
        ((1, 2, 30, 23, 10), 2, (1, 2, 4), (1, 3, 2), (1, 4, 3),
         (4, 1, 3), 2, None),
    ])
def test_conv(num_dims, input_size, out_channels, kernel_size, stride, padding,
              dilation, groups, bias):
    torch.manual_seed(42)

    stride = (1, ) * 3 if stride is None else stride
    padding = (0, ) * 3 if padding is None else padding
    dilation = (1, ) * 3 if dilation is None else dilation
    groups = 1 if groups is None else groups
    bias = True if bias is None else bias

    if num_dims != 3:
        kernel_size = kernel_size[:num_dims - 3]
        input_size = input_size[:num_dims - 3]
        stride = stride[:num_dims - 3]
        padding = padding[:num_dims - 3]
        dilation = dilation[:num_dims - 3]

    t = torch.randn(input_size)
    in_channels = t.size()[1]

    if num_dims == 1:
        conv_op = nn.Conv1d
    elif num_dims == 2:
        conv_op = nn.Conv2d
    elif num_dims == 3:
        conv_op = nn.Conv3d

    conv = conv_op(in_channels,
                   out_channels,
                   kernel_size,
                   stride,
                   padding,
                   dilation,
                   groups,
                   bias=bias)

    lin = nn.Linear(input_size[-1], input_size[-1])
    model = nn.Sequential(lin, conv)

    def training_step(x, l):
        model.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, l)
        loss.backward()
        # Check linear layer gradients to ensure that the conv
        # grad_input was calculated and backpropagated correctly
        rets = [out, lin.weight.grad, lin.bias.grad, conv.weight.grad]
        if bias:
            rets.append(conv.bias.grad)
        return rets

    label = torch.ones_like(conv(t))
    cpu_result = training_step(t, label)
    ipu_result = IPUContext(training_step, model=model)(t, label)

    for cpu, ipu in zip(cpu_result, ipu_result):
        helpers.assert_allclose(expected=cpu,
                                actual=ipu,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)
