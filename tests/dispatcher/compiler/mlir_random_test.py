#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

import torch

import poptorch
from poptorch.experimental import IPUContext

import helpers

# Set PyTorch's RNG seed to this manually.
rng_seed = 42

tensor_shapes = [
    # Hopefully large-enough tensor to get reasonable statistics.
    (3, 5, 10000)
]


# This is purely used for its __call__ to allow a more intuitive use of
# `rng_harness` like: rng_harness(my_fn, arg1, arg2=a2)(mean, std, torch.min),
# instead of having to mix up the function, its list of args and the list of
# stat functions in one list.
class StatChecker:
    def __init__(self, expect, actual):
        self.expect = expect
        self.actual = actual

    # Compare two sets of results statistically, using the given functions
    def __call__(self, *args):
        for tf in args:
            # pylint: disable=no-member
            helpers.assert_allclose(expected=tf(self.expect),
                                    actual=tf(self.actual),
                                    atol=1e-2,
                                    rtol=0.1)


# Helper, that takes a PyTorchy-function to run on the CPU & IPU.
#
# *args & **kwargs are forwarded straight to the given function.
#
# Returns a `StatChecker` class containing the results, that can be called with
# a list of statistical test functions to check the results for similarity.
def rng_harness(fn, *args, **kwargs):
    # Sometimes things need to be generated before calling into `rng_harness`,
    # and so the seed has already been set. In these cases, setting it again
    # can cause strange results.
    if torch.initial_seed() != rng_seed:
        torch.manual_seed(rng_seed)

    # Get the result from the CPU
    cpu_res = fn(*args, **kwargs)
    print(f"From CPU: {cpu_res}")

    ipu_res = IPUContext(fn)(*args, **kwargs)
    print(f"From IPU: {ipu_res}")

    # Return results in container, to later compare the results statistically
    return StatChecker(expect=cpu_res, actual=ipu_res)


# Helpers that are pluggable into `StatChecker`, that can get statistical
# values of a tensor of ints as well as floats.
def mean(inp):
    if inp.type() in [torch.float, torch.double]:
        return torch.mean(inp)
    return inp.double().mean()


def std(inp):
    if inp.type() in [torch.float, torch.double]:
        return torch.std(inp)
    return inp.double().std()


def var(inp):
    if inp.type() in [torch.float, torch.double]:
        return torch.var(inp)
    return inp.double().var()


# torch.randn
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_randn(shape):
    rng_harness(torch.randn, shape)(mean, std)


# torch.randn_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_randn_like(shape):
    torch.manual_seed(rng_seed)
    inp = torch.empty(shape)

    rng_harness(torch.randn_like, inp)(mean, std)


# torch.normal(float, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_float(shape):
    rng_harness(torch.normal, 5, 10, size=shape)(mean, std)


# torch.normal(Tensor, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_tensor(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdvs = torch.rand(shape) * 3

    rng_harness(torch.normal, means, stdvs)(mean, std)


# torch.normal(Tensor, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_float(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdv = 3

    rng_harness(torch.normal, means, stdv)(mean, std)


# torch.normal(float, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_float_tensor(shape):
    torch.manual_seed(rng_seed)
    desired_mean = 10
    stdvs = torch.rand(shape) * 3

    rng_harness(torch.normal, desired_mean, stdvs)(mean, std)


# torch.normal(Tensor, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_tensor_out(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdvs = torch.rand(shape) * 3

    def fn(means, stdvs):
        res = torch.empty(shape)
        torch.normal(means, stdvs, out=res)
        return res

    rng_harness(fn, means, stdvs)(mean, std)


# torch.normal(Tensor, float, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_tensor_float_out(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdv = 3

    def fn(means):
        res = torch.empty(shape)
        torch.normal(means, stdv, out=res)
        return res

    rng_harness(fn, means)(mean, std)


# torch.normal(float, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_float_tensor_out(shape):
    torch.manual_seed(rng_seed)
    desired_mean = 10
    stdvs = torch.rand(shape) * 3

    def fn(stdvs):
        res = torch.empty(shape)
        torch.normal(desired_mean, stdvs, out=res)
        return res

    rng_harness(fn, stdvs)(mean, std)


# torch.normal_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_normal_(shape):
    def fn():
        return torch.empty(shape).normal_(5, 10)

    rng_harness(fn)(torch.mean, torch.std)


# torch.rand
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_rand(shape):
    rng_harness(torch.rand, shape)(torch.min, torch.max, mean, var)


# torch.rand_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_rand_like(shape):
    torch.manual_seed(rng_seed)
    inp = torch.empty(shape)

    rng_harness(torch.rand_like, inp)(torch.min, torch.max, mean, var)


# torch.uniform_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_uniform_(shape):
    def fn():
        return torch.empty(shape).uniform_()

    rng_harness(fn)(torch.min, torch.max, mean, var)


# torch.exponential_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_exponential_(shape):
    def fn():
        return torch.empty(shape).exponential_()

    rng_harness(fn)(mean, std, var)


# torch.random_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype",
                         [torch.int, torch.float, torch.half, torch.bool])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_random_(shape, dtype):
    def fn():
        return torch.empty(shape, dtype=dtype).random_()

    rng_harness(fn)(mean, std)


# torch.random_(dtype=int8)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_random_int8(shape):
    # This is mainly to test boundaries of generated values.
    def fn():
        return torch.empty(shape, dtype=torch.int8).random_()

    rng_harness(fn)(torch.min, torch.max, mean, std)


# torch.random_(int, int)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype", [torch.int, torch.float, torch.half])
@pytest.mark.parametrize("limits", [(0, 1), (0, 2), (0, 3), (0, 5), (5, 500)])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_random_limits(shape, dtype, limits):
    def fn():
        return torch.empty(shape, dtype=dtype).random_(limits[0], limits[1])

    rng_harness(fn)(torch.min, torch.max, mean, std)


# torch.randint
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype", [torch.int, torch.float, torch.half])
@pytest.mark.parametrize("limits", [(0, 2), (0, 5), (5, 500)])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_randint(shape, dtype, limits):
    rng_harness(torch.randint, limits[0], limits[1], shape, dtype=dtype)\
               (torch.min, torch.max, mean, std)


# torch.randint_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype", [torch.int, torch.float])
@pytest.mark.parametrize("limits", [(2, 5)])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_randint_like(shape, dtype, limits):
    torch.manual_seed(42)
    inp = torch.empty(shape, dtype=dtype)

    rng_harness(torch.randint_like, inp, low=limits[0], high=limits[1])\
               (torch.min, torch.max, mean, std)


# torch.bernoulli_(float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_bernoulli_(shape, prob):
    def fn():
        return torch.empty(shape).bernoulli_(prob)

    rng_harness(fn)(mean)


# torch.bernoulli(Tensor, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_bernoulli_float(shape, prob):
    inp = torch.empty(shape)

    rng_harness(torch.bernoulli, inp, prob)(torch.mean)


# torch.bernoulli(Tensor, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_bernoulli_tensor(shape):
    torch.manual_seed(rng_seed)
    t = torch.rand(shape)

    rng_harness(torch.bernoulli, t)(mean)


# torch.bernoulli(Tensor, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="Your platform doesn't have MLIR support.")
def test_bernoulli_tensor_out(shape):
    torch.manual_seed(rng_seed)
    t = torch.rand(shape)

    def fn(seed):
        res = torch.empty(shape)
        torch.bernoulli(seed, out=res)
        return res

    rng_harness(fn, t)(mean)
