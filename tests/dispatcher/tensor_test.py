#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

import torch

import poptorch
from poptorch.experimental import IPUContext

import helpers

compilers = [poptorch.Compiler.PopART, poptorch.Compiler.MLIR]


@pytest.mark.parametrize("compiler", compilers)
@pytest.mark.mlirSupportRequired
def test_ipu_tensor_id(compiler):
    def f(x):
        x_tensor_id = poptorch.getIpuTensorId(x)
        assert x_tensor_id == poptorch.getIpuTensorId(
            x), "getIpuTensorId should be consistent between calls"

        y = torch.tensor([2.0, 3.0, 4.0],
                         device=helpers.outputDevice(),
                         dtype=torch.half)
        assert x_tensor_id != poptorch.getIpuTensorId(
            y), "Tensors should be different on the ipu"

        x = x + y
        assert x_tensor_id != poptorch.getIpuTensorId(
            x), "There should be a new tensor id after reassignment"

        return x

    torch.manual_seed(42)

    input = torch.rand(3, dtype=torch.half)

    IPUContext(f, compiler=compiler)(input)