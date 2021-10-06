#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import torch.nn as nn
import poptorch
import helpers


def test_simple_test():
    input = torch.ones([10])

    with poptorch.IPUScope([input]) as ipu:
        x = input + 5
        x = x * 3
        ipu.outputs([x])

    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu(input),
                            expected=torch.empty(10).fill_(18.0))


def test_simple_conv():
    input = torch.ones([1, 5, 25, 25])

    conv = nn.Conv2d(5, 10, 5)

    with poptorch.IPUScope([input], conv.named_parameters()) as ipu:
        x = conv(input)
        ipu.outputs([x])

    cpu = conv(input)
    ipu = ipu(input)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=cpu,
                            actual=ipu,
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)
