#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch

import pytest

# Pools
pool_operators = [
    torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
    torch.nn.MaxUnpool1d, torch.nn.MaxUnpool2d, torch.nn.MaxUnpool3d,
    torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d,
    torch.nn.FractionalMaxPool2d, torch.nn.LPPool1d, torch.nn.LPPool2d,
    torch.nn.AdaptiveMaxPool1d, torch.nn.AdaptiveMaxPool2d,
    torch.nn.AdaptiveMaxPool3d, torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d
]

# Supported.
pool_1D = [torch.nn.MaxPool1d, torch.nn.AvgPool1d]
pool_2D = [torch.nn.MaxPool2d, torch.nn.AvgPool2d]
pool_3D = [torch.nn.MaxPool3d, torch.nn.AvgPool3d]
adaptive_pool_2D = [
    torch.nn.AdaptiveAvgPool2d
]  #, torch.nn.AdaptiveMaxPool2d] # Adaptive max pooling isn't supported due to returning 2 outputs, easy fix.
# TODO (T22978)


def execute_and_check_wrapper(model, input, check_shape_only=False):
    # Run on CPU.
    nativeOut = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    print(nativeOut.size())

    if not check_shape_only:
        torch.testing.assert_allclose(poptorch_out, nativeOut)
    else:
        # This is due to adaptive pooling's process essentially being an implementation detail.
        assert poptorch_out.size() == nativeOut.size()


@pytest.mark.parametrize("op", pool_2D)
def test_pool2D(op):

    torch.manual_seed(42)

    input = torch.randn(20, 16, 50, 10)

    # pool of square window of size=3, stride=2
    model = op(3, stride=2)
    execute_and_check_wrapper(model, input)

    # pool of square window of size=3, stride=2, ceil_mode=True
    model = op(3, stride=2, ceil_mode=True)
    execute_and_check_wrapper(model, input)

    #  pool of non-square window
    model = op((3, 2), stride=(2, 1))
    execute_and_check_wrapper(model, input)

    # pool of square window of size=3, stride=2, padding=1
    model = op(3, stride=2, padding=1)
    execute_and_check_wrapper(model, input)

    if op == torch.nn.AvgPool2d:
        # pool of square window of size=3, stride=2, padding=1, pool excludes padding
        model = op(3, stride=2, padding=1, count_include_pad=False)
        execute_and_check_wrapper(model, input)


@pytest.mark.parametrize("op", adaptive_pool_2D)
def test_adaptive_pool2D(op):

    torch.manual_seed(42)

    input = torch.randn(20, 16, 50, 10)

    # Target size of (40, 5)
    model = op((40, 5))

    # Only check shape due to exact pytorch process being an implementaion detail.
    execute_and_check_wrapper(model, input, check_shape_only=True)


# Padding

one_d_pads = [
    torch.nn.ReflectionPad1d, torch.nn.ReplicationPad1d, torch.nn.ConstantPad1d
]


@pytest.mark.parametrize("op", one_d_pads)
def test_1D_pads(op):
    torch.manual_seed(42)

    # torch.nn.ConstantPad1d, 'torch.nn.ConstantPad2d', 'torch.nn.ConstantPad3d',
    # One D case
    oneDTensor = torch.randn(1, 2, 4)

    # Pad evenly in both directions.

    if op == torch.nn.ConstantPad1d:
        model = op(2, 4.7)
    else:
        model = op(3)
    execute_and_check_wrapper(model, oneDTensor)

    # Pad unevenly in both directions.
    if op == torch.nn.ConstantPad1d:
        model = op((3, 2), 0.12456)
    else:
        model = op((3, 2))
    execute_and_check_wrapper(model, oneDTensor)


two_d_pads = [
    torch.nn.ReflectionPad2d, torch.nn.ReplicationPad2d,
    torch.nn.ConstantPad2d, torch.nn.ZeroPad2d
]


@pytest.mark.parametrize("op", two_d_pads)
def test_2D_pads(op):
    # 2D Case
    twoDTensor = torch.randn(1, 2, 4, 4)

    # Pad evenly in all directions.

    if op == torch.nn.ConstantPad2d:
        model = op(6, 2.3)
    else:
        model = op(2)
    execute_and_check_wrapper(model, twoDTensor)

    # Pad unevenly in all directions.
    if op == torch.nn.ConstantPad2d:
        model = op((3, 2, 1, 5), 4.7)
    else:
        model = op((3, 2, 1, 3))

    execute_and_check_wrapper(model, twoDTensor)


three_d_pads = [torch.nn.ReplicationPad3d, torch.nn.ConstantPad3d]


@pytest.mark.parametrize("op", three_d_pads)
def test_3D_pads(op):
    # 3D Case
    threeDTensor = torch.randn(1, 2, 4, 4, 4)

    # Pad evenly in all directions.
    if op == torch.nn.ConstantPad3d:
        model = op(2, 6.4)
    else:
        model = op(3)
    execute_and_check_wrapper(model, threeDTensor)

    # Pad unevenly in all directions.
    if op == torch.nn.ConstantPad3d:
        model = op((3, 2, 1, 5, 3, 4), 7.2)
    else:
        model = op((3, 2, 1, 5, 3, 4))
    execute_and_check_wrapper(model, threeDTensor)
