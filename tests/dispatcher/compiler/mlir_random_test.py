#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

import torch

import poptorch
from poptorch.enums import Compiler
from poptorch.experimental import IPUContext, IPUScope

import helpers

tensor_shapes = [
    # Hopefully large-enough tensor to get reasonable statistics.
    (3, 5, 10000)
]


# Compare two sets of results statistically, using the given functions
def check_stats(expect_base, actual_base, test_fns):
    for tf in test_fns:
        # pylint: disable=no-member
        helpers.assert_allclose(expected=tf(expect_base),
                                actual=tf(actual_base),
                                atol=1e-2,
                                rtol=0.1)


# Helper, that takes a PyTorchy-function to run on the CPU & IPU, and some
# statistical test functions to check the results for similarity.
def rng_harness(fn, test_fns):
    torch.manual_seed(42)

    # Get the result from the CPU
    cpu_res = fn()
    print(f"From CPU: {cpu_res}")

    ipu_res = IPUContext(fn)()
    print(f"From IPU: {ipu_res}")

    # Compare the CPU & IPU results statistically
    check_stats(cpu_res, ipu_res, test_fns)


# Helpers that are pluggable into `check_stats`, that can get the mean & std of
# a tensor of ints as well as floats.
def mean(inp):
    if inp.type() in [torch.float, torch.double]:
        return torch.mean(inp)
    return inp.double().mean()


def std(inp):
    if inp.type() in [torch.float, torch.double]:
        return torch.std(inp)
    return inp.double().std()


# torch.randn
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_randn(shape):
    def rand():
        return torch.randn(shape)

    rng_harness(rand, [torch.mean, torch.std])


# torch.randn_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_randn_like(shape):
    torch.manual_seed(42)
    inp = torch.empty(shape)

    cpu_res = torch.randn_like(inp)

    ipu_res = IPUContext(torch.randn_like)(inp)

    check_stats(cpu_res, ipu_res, [torch.mean, torch.std])


# torch.normal(float, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_float(shape):
    def rand():
        return torch.normal(5, 10, size=shape)

    rng_harness(rand, [torch.mean, torch.std])


# torch.normal(Tensor, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_tensor(shape):
    torch.manual_seed(42)
    means = torch.rand(shape) * 10
    stdvs = torch.rand(shape) * 3

    cpu_res = torch.normal(means, stdvs)

    ipu_res = IPUContext(torch.normal)(means, stdvs)

    check_stats(cpu_res, ipu_res, [torch.mean, torch.std])


# torch.normal(Tensor, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_float(shape):
    torch.manual_seed(42)
    means = torch.rand(shape) * 10
    stdv = 3

    cpu_res = torch.normal(means, stdv)

    ipu_res = IPUContext(torch.normal)(means, stdv)

    check_stats(cpu_res, ipu_res, [torch.mean, torch.std])


# torch.normal(float, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_float_tensor(shape):
    torch.manual_seed(42)
    mean = 10
    stdvs = torch.rand(shape) * 3

    cpu_res = torch.normal(mean, stdvs)

    ipu_res = IPUContext(torch.normal)(mean, stdvs)

    check_stats(cpu_res, ipu_res, [torch.mean, torch.std])


# torch.normal(Tensor, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_tensor_out(shape):
    torch.manual_seed(42)
    means = torch.rand(shape) * 10
    stdvs = torch.rand(shape) * 3

    cpu_res = torch.empty(shape)
    torch.normal(means, stdvs, out=cpu_res)

    with IPUScope([means, stdvs], compile_using=Compiler.MLIR) as ipu:
        res = torch.empty(shape)
        torch.normal(means, stdvs, out=res)
        ipu.outputs([res])
    ipu_res = ipu(means, stdvs)

    check_stats(cpu_res, ipu_res, [torch.mean, torch.std])


# torch.normal(Tensor, float, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_float_out(shape):
    torch.manual_seed(42)
    means = torch.rand(shape) * 10
    stdv = 3

    cpu_res = torch.empty(shape)
    torch.normal(means, stdv, out=cpu_res)

    with IPUScope([means], compile_using=Compiler.MLIR) as ipu:
        res = torch.empty(shape)
        torch.normal(means, stdv, out=res)
        ipu.outputs([res])
    ipu_res = ipu(means)

    check_stats(cpu_res, ipu_res, [torch.mean, torch.std])


# torch.normal(float, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_float_tensor_out(shape):
    torch.manual_seed(42)
    mean = 10
    stdvs = torch.rand(shape) * 3

    cpu_res = torch.empty(shape)
    torch.normal(mean, stdvs, out=cpu_res)

    with IPUScope([stdvs], compile_using=Compiler.MLIR) as ipu:
        res = torch.empty(shape)
        torch.normal(mean, stdvs, out=res)
        ipu.outputs([res])
    ipu_res = ipu(stdvs)

    check_stats(cpu_res, ipu_res, [torch.mean, torch.std])


# torch.normal_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_(shape):
    def rand():
        return torch.empty(shape).normal_(5, 10)

    rng_harness(rand, [torch.mean, torch.std])


# torch.rand
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_rand(shape):
    def rand():
        return torch.rand(shape)

    rng_harness(rand, [torch.min, torch.max, torch.mean, torch.var])


# torch.rand_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_rand_like(shape):
    torch.manual_seed(42)
    inp = torch.empty(shape)

    cpu_res = torch.rand_like(inp)

    ipu_res = IPUContext(torch.rand_like)(inp)

    check_stats(cpu_res, ipu_res,
                [torch.min, torch.max, torch.mean, torch.var])


# torch.uniform_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_uniform_(shape):
    def rand():
        return torch.empty(shape).uniform_()

    rng_harness(rand, [torch.min, torch.max, torch.mean, torch.var])
