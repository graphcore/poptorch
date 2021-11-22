#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch
from poptorch import enums


def cat_stack_harness(op, dim, dtype):
    t1 = torch.ones(3, 2, dtype=dtype)
    t2 = torch.zeros(3, 2, dtype=dtype)

    torch_out = op((t1, t2), dim)

    with poptorch.IPUScope([t1, t2], compile_using=enums.Compiler.MLIR) as ipu:
        out = op((t1, t2), dim)
        ipu.outputs([out])

    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu(t1, t2), expected=torch_out)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize(
    "params",
    [
        # dim, dtype, alias
        (0, torch.float32, torch.cat),
        (1, torch.int32, torch.concat),
    ])
def test_cat(params):
    dim, dtype, alias = params
    cat_stack_harness(alias, dim, dtype)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7.3 is not currently supported in MLIR.")
@pytest.mark.parametrize(
    "params",
    [
        # dim, dtype
        (0, torch.float32),
        (1, torch.int32),
    ])
def test_stack(params):
    dim, dtype = params
    cat_stack_harness(torch.stack, dim, dtype)
