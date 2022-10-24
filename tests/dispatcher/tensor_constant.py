#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import torch
import helpers
import poptorch
from poptorch.experimental import IPUContext

compilers = [poptorch.Compiler.PopART, poptorch.Compiler.MLIR]


@pytest.mark.parametrize("compiler", compilers)
@pytest.mark.parametrize("dtype", [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
])
def test_tensor_constant_ints(compiler, dtype):
    def f(x):
        x = x.to(helpers.outputDevice())
        return x + torch.tensor(
            [2, 3, 4], device=helpers.outputDevice(), dtype=dtype)

    torch.manual_seed(42)

    input = torch.rand(3, dtype=torch.half)

    cpu = f(input)
    ipu = IPUContext(f, compiler=compiler)(input)

    helpers.assert_allequal(expected=cpu, actual=ipu)


@pytest.mark.parametrize("compiler", compilers)
@pytest.mark.parametrize("dtype", [torch.half, torch.float, torch.double])
def test_tensor_constant_floats(compiler, dtype):
    def f(x):
        x = x.to(helpers.outputDevice())
        return x + torch.tensor(
            [2, 3, 4], device=helpers.outputDevice(), dtype=dtype)

    torch.manual_seed(42)

    input = torch.rand(3, dtype=torch.half)

    cpu = f(input)
    ipu = IPUContext(f, compiler=compiler)(input)

    # (For PopART) Allow minor changes in floating-point-based values; for f16,
    # 1-bit can mean a magnitude 1e-2 change depending on value.
    atol = 1e-2 if dtype == torch.half else None

    helpers.assert_allclose(expected=cpu, actual=ipu, atol=atol)


@pytest.mark.parametrize("compiler", compilers)
def test_tensor_constant_bool(compiler):
    def f(x):
        x = x.to(helpers.outputDevice())
        return torch.logical_and(
            x, torch.tensor([False, True, False],
                            device=helpers.outputDevice()))

    input = torch.tensor([False, True, True])

    cpu = f(input)
    ipu = IPUContext(f, compiler=compiler)(input)

    helpers.assert_allequal(expected=cpu, actual=ipu)
