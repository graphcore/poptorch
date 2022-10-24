#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import ctypes
import pathlib

import pytest

import torch
import poptorch

# Test that the `grad_fn` is set correctly for poptorch ops.
#
# for the PopART backend, since PopART does its own thing in the backward pass
# we need only check that there's *a* `grad_fn`, to appease PyTorch.

size = 5

myso = list(pathlib.Path("tests").rglob("libcustom_cube_op.*"))
assert myso, "Failed to find libcustom_cube_op"
myop = ctypes.cdll.LoadLibrary(myso[0])


# pylint: disable=unused-argument
def attribute(self, x):
    with poptorch.Attribute(__outline={"layer": "embedding"}):
        x = torch.nn.functional.group_norm(x, 1)
    return x


def available_memory(self, x):
    return poptorch.set_available_memory(x, 0.5)


def block(self, x):
    poptorch.Block.useAutoId()
    with poptorch.Block(ipu_id=0):
        x = x * 4
    with poptorch.Block(ipu_id=1):
        x = x * 2
    return x


def cpu_op(self, x):
    return self.cpu(x)


def custom_op(self, x):
    return poptorch.custom_op([x, x],
                              "Cube",
                              "com.acme",
                              1,
                              example_outputs=[x, x])


def dynamic_slice(self, x):
    return poptorch.dynamic_slice(x, 0, torch.tensor([0]), size, 1)


def for_loop(self, x):
    return poptorch.for_loop(1, lambda x: x + 2, [x])


def identity_loss(self, x):
    return poptorch.identity_loss(x, "sum")


def ipu_print_tensor(self, x):
    return poptorch.ipu_print_tensor(x)


def multiconv(self, x):
    with poptorch.MultiConv():
        # Easier to make new inputs, so make sure at least 1 requires grad.
        a = torch.randn(1, 4, 5, 5, requires_grad=True)
        b = torch.randn(8, 4, 3, 3)
        return torch.nn.functional.conv2d(a, b, padding=1)


def name_scope(self, x):
    with poptorch.NameScope("hello"):
        return x * 2


def nop(self, x):
    return poptorch.nop(x)


def recomputation_checkpoint(self, x):
    return poptorch.recomputationCheckpoint([x])


def serialized_matmul(self, x):
    return poptorch.serializedMatMul(self.weight, x,
                                     poptorch.MatMulSerializationMode.Disabled)


# pylint: enable=unused-argument

poptorch_fns = [
    attribute,
    available_memory,
    block,
    cpu_op,
    custom_op,
    dynamic_slice,
    for_loop,
    identity_loss,
    ipu_print_tensor,
    multiconv,
    name_scope,
    nop,
    recomputation_checkpoint,
    serialized_matmul,
]


def check_grad_fn(x):
    if isinstance(x, (list, tuple)):
        for t in x:
            assert not t.is_leaf
    else:
        assert not x.is_leaf


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "weight",
            torch.nn.Parameter(torch.rand(size, size), requires_grad=True))

        self.cpu_op_body = lambda x: x * 5
        self.cpu = poptorch.CPU(self.cpu_op_body, "MyCPUOp")

        self.fn = None

    def forward(self, x):
        x = torch.mm(self.weight, x)  # give `x` a `grad_fn`

        x = self.fn(self, x)

        check_grad_fn(x)

        return x


@pytest.mark.parametrize("fn", poptorch_fns)
def test_poptorch_op(fn):
    model = Model()

    model.fn = fn

    # Only really needed for `block`
    opts = poptorch.Options()
    opts.deviceIterations(2)

    model = poptorch.inferenceModel(model, opts)

    model(torch.rand(size * opts.device_iterations, 1))


# for poptorch.set_overlap_for_input, its argument must be a direct graph input
# so we can't use the more general handler.
def test_overlap():
    class Model(torch.nn.Module):
        def forward(self, x):
            x.requires_grad_(True)

            x = poptorch.set_overlap_for_input(x,
                                               poptorch.OverlapMode.NoOverlap)
            check_grad_fn(x)
            x = x + 1
            x = poptorch.set_overlap_for_output(x,
                                                poptorch.OverlapMode.NoOverlap)
            check_grad_fn(x)

            return x

    model = poptorch.inferenceModel(Model())

    model(torch.rand(size, 1))
