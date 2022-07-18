#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext

SUPPORTED_TYPES = (torch.float16, torch.float32, torch.int32)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("cast_from", SUPPORTED_TYPES)
@pytest.mark.parametrize("cast_to", SUPPORTED_TYPES)
def test_cast(cast_from, cast_to):
    t_in = torch.randint(low=0, high=10, size=(10, ), dtype=cast_from)

    def cast_op(t):
        return t.to(cast_to)

    ipu_result = IPUContext(cast_op)(t_in)
    cpu_result = cast_op(t_in)

    helpers.assert_allequal(actual=ipu_result, expected=cpu_result)


def cat_stack_harness(op, dim, dtype):
    t1 = torch.ones(3, 2, dtype=dtype)
    t2 = torch.zeros(3, 2, dtype=dtype)

    ipu_result = IPUContext(op)((t1, t2), dim)
    cpu_result = op((t1, t2), dim)

    helpers.assert_allequal(actual=ipu_result, expected=cpu_result)


@pytest.mark.mlirSupportRequired
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


@pytest.mark.mlirSupportRequired
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


@pytest.mark.mlirSupportRequired
def test_where():
    zeros = torch.zeros(3, 3)
    ones = torch.ones(3, 3)
    cond = torch.tensor([[True, False, True], [False, True, True],
                         [False, False, True]])

    ipu_result = IPUContext(torch.where)(cond, zeros, ones)
    cpu_result = torch.where(cond, zeros, ones)

    helpers.assert_allequal(actual=ipu_result, expected=cpu_result)


# Check the canonicalizer for copy_ is not too eager with its optimisations
@pytest.mark.mlirSupportRequired
def test_copy_optim():
    def fn(input, t2):
        a = torch.empty([3, 3], device=helpers.outputDevice())
        a.copy_(input)
        input.copy_(t2)
        return a, input

    t = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 7, 8]]).float()
    t2 = torch.ones(3, 3)
    ipu_result = IPUContext(fn)(t, t2)
    cpu_result = fn(t, t2)

    for ipu_out, cpu_out in zip(ipu_result, cpu_result):
        print(f"ipu {ipu_out} cpu {cpu_out}", flush=True)
        helpers.assert_allclose(actual=ipu_out, expected=cpu_out)


@pytest.mark.mlirSupportRequired
def test_as_strided():
    def op_harness(op, *args):
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 7, 8]])

        cpu_result = op(t, *args)
        ipu_result = IPUContext(op)(t, *args)

        helpers.assert_allequal(actual=ipu_result, expected=cpu_result)

    op_harness(torch.as_strided, [3, 3], [3, 1])

    err_msg = (
        r"Poptorch does not support arbitrary manipulations of the shape and "
        r"stride of a tensor\. Prefer other view functions like "
        r"torch\.tensor\.expand\(\) over setting the shape and stride of a "
        r"view manually\..*")
    with pytest.raises(poptorch.Error, match=err_msg):
        op_harness(torch.as_strided, [3, 3], [1, 3])


@pytest.mark.mlirSupportRequired
def expand_reshape_view_harness(in_shape, new_shape, op):
    torch.manual_seed(42)

    t = torch.randn(in_shape)

    def f(t):
        if op == "broadcast_to":
            return torch.broadcast_to(t, (new_shape)), t
        if op == "expand":
            return t.expand(new_shape), t
        if op == "reshape":
            return t.reshape(new_shape), t
        if op == "view":
            return t.view(new_shape), t
        raise ValueError("Invalid op")

    try:
        cpu_result = f(t)
    except RuntimeError:
        # Let any exception come from IPUScope only
        cpu_result = None

    ipu_result = IPUContext(f)(t)

    helpers.assert_allequal(actual=ipu_result[0], expected=cpu_result[0])

    # Nb this is not guaranteed in all cases e.g. strides on CPU
    helpers.assert_allequal(actual=ipu_result[1], expected=t)


@pytest.mark.mlirSupportRequired
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
    with pytest.raises((poptorch.Error, RuntimeError), match=err_msg):
        expand_reshape_view_harness((2, 4), (1, 7), op)

    err_msg = ("only one dimension can be inferred")
    with pytest.raises((poptorch.Error, RuntimeError), match=err_msg):
        expand_reshape_view_harness((7, 4), (-1, -1), op)

    with pytest.raises((poptorch.Error, RuntimeError), match=err_msg):
        expand_reshape_view_harness((3, 4), (-1, -1, 2), op)


@pytest.mark.mlirSupportRequired
def test_reshape_sparse():
    s = torch.sparse_coo_tensor([(0, 1), (2, 2), (2, 3)], [1.0, 2.0],
                                (2, 3, 4))

    err_msg = (r"You cannot pass sparse tensors as inputs to IPUScope\.")

    def reshape_sparse(t):
        return t.reshape((6, 4))

    with pytest.raises(ValueError, match=err_msg):
        IPUContext(reshape_sparse)(s)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 4),  # standard
        (2, 2, 2, 4, 4),  # more dimensions
        (2, 4, -1),  # negative dimension
        (2, 2, -1, 2, 4),  # negative & extra dimensions
    ])
@pytest.mark.parametrize("op", ["broadcast_to", "expand"])
def test_expand(shape, op):
    expand_reshape_view_harness((2, 1, 4), shape, op)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("shape", [(5), (1, 2, 3)])
def test_expand_scalar(shape):
    expand_reshape_view_harness([], shape, "expand")


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize(
    "case",
    [((2, 4), r"should have at least as many dimensions"),
     ((2, 2, 5), r"Can only expand dimensions of size 1"),
     ((2, -1, 2, 4, 4), r"tried to set an added dimension's length to -1")])
def test_expand_error(case):
    shape, msg = case
    with pytest.raises(poptorch.Error, match=msg):
        expand_reshape_view_harness((2, 1, 4), shape, "expand")


# Simple harness to test that the given `fn` returns a view of its input tensor,
# not just a copy.
#
# This is tested by modifying the source tensor after the view has been made,
# and checking that this change is reflected in the view.
#
# :param in_shape: shape of tensor to give to test function.
# :param fn: function of signature `fn(input_tensor, ...) -> output_tensor`.
def is_view_harness(in_shape, fn, *args, **kwargs):
    num_elems = 1
    for d in in_shape:
        num_elems *= d

    t_cpu = torch.linspace(1, num_elems, num_elems).view(in_shape)
    t_ipu = torch.empty(in_shape).copy_(t_cpu)

    def is_view_fn(t, *args, **kwargs):
        res = fn(t, *args, **kwargs)
        t.add_(8.0)
        return res

    ipu_res = IPUContext(is_view_fn)(t_ipu, *args, **kwargs)
    cpu_res = is_view_fn(t_cpu, *args, **kwargs)

    helpers.assert_allequal(actual=ipu_res, expected=cpu_res)


@pytest.mark.mlirSupportRequired
def test_view_is_view():
    shape = (3, 4, 5)
    view_shape = shape[::-1]

    fn = lambda t, s: t.view(s)

    is_view_harness(shape, fn, view_shape)


@pytest.mark.mlirSupportRequired
def test_expand_is_view():
    shape = (3, 1, 5)
    expanded_shape = (3, 4, 5)

    fn = lambda t, s: t.expand(s)

    is_view_harness(shape, fn, expanded_shape)
