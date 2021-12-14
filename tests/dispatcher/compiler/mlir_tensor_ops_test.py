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
                    reason="CentOS 7 is not currently supported in MLIR.")
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
                    reason="CentOS 7 is not currently supported in MLIR.")
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


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_where():
    zeros = torch.zeros(3, 3)
    ones = torch.ones(3, 3)
    cond = torch.tensor([[True, False, True], [False, True, True],
                         [False, False, True]])
    torch_out = torch.where(cond, zeros, ones)

    with poptorch.IPUScope([zeros, ones, cond],
                           compile_using=enums.Compiler.MLIR) as ipu:
        ipu_out = torch.where(cond, zeros, ones)
        ipu.outputs([ipu_out])

    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu(zeros, ones, cond), expected=torch_out)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def expand_reshape_view_harness(in_shape, new_shape, op):
    torch.manual_seed(42)

    t = torch.randn(in_shape)

    try:
        if op == "expand":
            torch_out = t.expand(new_shape)
        elif op == "reshape":
            torch_out = t.reshape(new_shape)
        elif op == "view":
            torch_out = t.view(new_shape)
        else:
            raise ValueError("Invalid op")
    except RuntimeError:
        # Let any exception come from IPUScope only
        torch_out = None

    with poptorch.IPUScope([t], compile_using=enums.Compiler.MLIR) as ipu:
        if op == "expand":
            out = t.expand(new_shape)
        elif op == "reshape":
            out = t.reshape(new_shape)
        elif op == "view":
            out = t.view(new_shape)
        else:
            raise ValueError("Invalid op")
        ipu.outputs([out, t])

    ipu_out = ipu(t)
    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu_out[0], expected=torch_out)

    # Nb this is not guaranteed in all cases e.g. strides on CPU
    # pylint: disable=no-member
    helpers.assert_allequal(actual=ipu_out[1], expected=t)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("op", ["reshape", "view"])
def test_reshape_view(op):
    # Ordinary reshape/view
    expand_reshape_view_harness((2, 3, 4), (4, 2, 3), op)

    # Reshape/view with -1 dim
    expand_reshape_view_harness((2, 3, 4), (-1, 2, 3), op)
    expand_reshape_view_harness((2, 3, 4), (4, -1, 3), op)
    expand_reshape_view_harness((2, 3, 4), (2, 2, -1), op)

    # Error conditions shape\'\[1, 7\]\'
    err_msg = (r"\]\' is invalid for input of size 8")
    with pytest.raises(RuntimeError, match=err_msg):
        expand_reshape_view_harness((2, 4), (1, 7), op)

    err_msg = ("only one dimension can be inferred")
    with pytest.raises(RuntimeError, match=err_msg):
        expand_reshape_view_harness((7, 4), (-1, -1), op)

    with pytest.raises(RuntimeError, match=err_msg):
        expand_reshape_view_harness((3, 4), (-1, -1, 2), op)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_reshape_sparse():
    s = torch.sparse_coo_tensor([(0, 1), (2, 2), (2, 3)], [1.0, 2.0],
                                (2, 3, 4))

    err_msg = (r"You cannot pass sparse tensors as inputs to " +
               r"poptorch\.IPUScope\.")
    with pytest.raises(ValueError, match=err_msg):
        with poptorch.IPUScope([s], compile_using=enums.Compiler.MLIR) as ipu:
            out = s.reshape((6, 4))
            ipu.outputs([out])


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_expand():
    expand_reshape_view_harness((2, 1, 4), (2, 4, 4), "expand")
    expand_reshape_view_harness((2, 1, 4), (2, 2, 2, 4, 4), "expand")

    # TODO T52507 Fully inplement expand
    with pytest.raises(RuntimeError):
        expand_reshape_view_harness((2, 1, 4), (2, 4, -1), "expand")
