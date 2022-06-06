#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

import torch

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

    # Get results from CPU & IPU
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
@pytest.mark.mlirSupportRequired
def test_randn(shape):
    def fn(shape):
        return torch.randn(shape, device=helpers.outputDevice())

    rng_harness(fn, shape)(mean, std)


# torch.randn_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_randn_like(shape):
    torch.manual_seed(rng_seed)
    inp = torch.empty(shape)

    rng_harness(torch.randn_like, inp)(mean, std)


# torch.normal(float, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_float(shape):
    def fn(mean_val, std_val, size):
        return torch.normal(mean_val,
                            std_val,
                            size,
                            device=helpers.outputDevice())

    rng_harness(fn, 5, 10, size=shape)(mean, std)


# torch.normal(Tensor, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_tensor_tensor(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdvs = torch.rand(shape) * 3

    rng_harness(torch.normal, means, stdvs)(mean, std)


# torch.normal(Tensor, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_tensor_float(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdv = 3

    rng_harness(torch.normal, means, stdv)(mean, std)


# torch.normal(float, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_float_tensor(shape):
    torch.manual_seed(rng_seed)
    desired_mean = 10
    stdvs = torch.rand(shape) * 3

    rng_harness(torch.normal, desired_mean, stdvs)(mean, std)


# torch.normal(Tensor, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_tensor_tensor_out(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdvs = torch.rand(shape) * 3

    def fn(means, stdvs):
        res = torch.empty(shape, device=helpers.outputDevice())
        torch.normal(means, stdvs, out=res)
        return res

    rng_harness(fn, means, stdvs)(mean, std)


# torch.normal(Tensor, float, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_tensor_float_out(shape):
    torch.manual_seed(rng_seed)
    means = torch.rand(shape) * 10
    stdv = 3

    def fn(means):
        res = torch.empty(shape, device=helpers.outputDevice())
        torch.normal(means, stdv, out=res)
        return res

    rng_harness(fn, means)(mean, std)


# torch.normal(float, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_float_tensor_out(shape):
    torch.manual_seed(rng_seed)
    desired_mean = 10
    stdvs = torch.rand(shape) * 3

    def fn(stdvs):
        res = torch.empty(shape, device=helpers.outputDevice())
        torch.normal(desired_mean, stdvs, out=res)
        return res

    rng_harness(fn, stdvs)(mean, std)


# torch.normal_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_normal_(shape):
    def fn():
        return torch.empty(shape, device=helpers.outputDevice()).normal_(5, 10)

    rng_harness(fn)(torch.mean, torch.std)


# torch.rand
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_rand(shape):
    def fn(shape):
        return torch.rand(shape, device=helpers.outputDevice())

    rng_harness(fn, shape)(torch.min, torch.max, mean, var)


# torch.rand_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_rand_like(shape):
    torch.manual_seed(rng_seed)
    inp = torch.empty(shape)

    rng_harness(torch.rand_like, inp)(torch.min, torch.max, mean, var)


# torch.uniform_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_uniform_(shape):
    def fn():
        return torch.empty(shape, device=helpers.outputDevice()).uniform_()

    rng_harness(fn)(torch.min, torch.max, mean, var)


# torch.exponential_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_exponential_(shape):
    def fn():
        return torch.empty(shape, device=helpers.outputDevice()).exponential_()

    rng_harness(fn)(mean, std, var)


# torch.exponential_
@pytest.mark.mlirSupportRequired
def test_exponential_inf():
    # Hopefully 5e7 is enough to generate the boundaries. There isn't enough
    # tile memory to set this much higher
    def fn():
        return torch.torch.empty((int(5e7)),
                                 device=helpers.outputDevice()).exponential_()

    ipu_res = IPUContext(fn)()

    assert not torch.isinf(ipu_res).any()


# torch.random_
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype",
                         [torch.int, torch.float, torch.half, torch.bool])
@pytest.mark.mlirSupportRequired
def test_random_(shape, dtype):
    def fn():
        return torch.empty(shape, dtype=dtype,
                           device=helpers.outputDevice()).random_()

    rng_harness(fn)(mean, std)


# torch.random_(dtype=int8)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_random_int8(shape):
    # This is mainly to test boundaries of generated values.
    def fn():
        return torch.empty(shape,
                           dtype=torch.int8,
                           device=helpers.outputDevice()).random_()

    rng_harness(fn)(torch.min, torch.max, mean, std)


# torch.random_(int, int)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype", [torch.int, torch.float, torch.half])
@pytest.mark.parametrize("limits", [(0, 1), (0, 2), (0, 3), (0, 5), (5, 500)])
@pytest.mark.mlirSupportRequired
def test_random_limits(shape, dtype, limits):
    def fn():
        return torch.empty(shape, dtype=dtype,
                           device=helpers.outputDevice()).random_(
                               limits[0], limits[1])

    rng_harness(fn)(torch.min, torch.max, mean, std)


# torch.randint
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype", [torch.int, torch.float, torch.half])
@pytest.mark.parametrize("limits", [(0, 2), (0, 5), (5, 500)])
@pytest.mark.mlirSupportRequired
def test_randint(shape, dtype, limits):
    def fn(*args, **kwargs):
        return torch.randint(*args, **kwargs, device=helpers.outputDevice())

    rng_harness(fn, limits[0], limits[1], shape, dtype=dtype)\
               (torch.min, torch.max, mean, std)


# torch.randint_like
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("dtype", [torch.int, torch.float])
@pytest.mark.parametrize("limits", [(2, 5)])
@pytest.mark.mlirSupportRequired
def test_randint_like(shape, dtype, limits):
    torch.manual_seed(42)
    inp = torch.empty(shape, dtype=dtype)

    rng_harness(torch.randint_like, inp, low=limits[0], high=limits[1])\
               (torch.min, torch.max, mean, std)


# torch.bernoulli_(float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.mlirSupportRequired
def test_bernoulli_(shape, prob):
    def fn():
        return torch.empty(shape,
                           device=helpers.outputDevice()).bernoulli_(prob)

    rng_harness(fn)(mean)


# torch.bernoulli(Tensor, float)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
@pytest.mark.mlirSupportRequired
def test_bernoulli_float(shape, prob):
    inp = torch.empty(shape)

    rng_harness(torch.bernoulli, inp, prob)(torch.mean)


# torch.bernoulli(Tensor, Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_bernoulli_tensor(shape):
    torch.manual_seed(rng_seed)
    t = torch.rand(shape)

    rng_harness(torch.bernoulli, t)(mean)


# torch.bernoulli(Tensor, Tensor, out=Tensor)
@pytest.mark.parametrize("shape", tensor_shapes)
@pytest.mark.mlirSupportRequired
def test_bernoulli_tensor_out(shape):
    torch.manual_seed(rng_seed)
    t = torch.rand(shape)

    def fn(seed):
        res = torch.empty(shape, device=helpers.outputDevice())
        torch.bernoulli(seed, out=res)
        return res

    rng_harness(fn, t)(mean)


# Need something different to rng_harness for randperm, for custom checking.
def randperm_harness(fn, n, dtype):
    res = IPUContext(fn)()
    print(f"From IPU: {res}")

    # Check correct result dtype
    desired_dtype = dtype
    if desired_dtype == torch.int64:
        desired_dtype = torch.int32
    if desired_dtype == torch.double:
        desired_dtype = torch.float
    assert res.dtype == desired_dtype

    # Check correct values
    expected_values = list(range(n))
    for val in res:
        assert val in expected_values,\
            f"Result contained a value it shouldn't have ({val})"
        expected_values.remove(val)

    assert len(expected_values) == 0,\
        f"Result didn't contain these values it should have: {expected_values}"


# torch.randperm
@pytest.mark.parametrize("n", [0, 1, 5, 10])
@pytest.mark.parametrize("dtype", [
    torch.int16, torch.int32, torch.int64, torch.half, torch.float,
    torch.double
])
@pytest.mark.mlirSupportRequired
def test_randperm(n, dtype):
    torch.manual_seed(rng_seed)

    def fn():
        return torch.randperm(n, dtype=dtype, device=helpers.outputDevice())

    randperm_harness(fn, n, dtype)


# torch.randperm_out
@pytest.mark.parametrize("n", [5])
@pytest.mark.parametrize("dtype", [
    torch.int16, torch.int32, torch.int64, torch.half, torch.float,
    torch.double
])
@pytest.mark.mlirSupportRequired
def test_randperm_out(n, dtype):
    torch.manual_seed(rng_seed)

    def fn():
        res = torch.empty((n, ), dtype=dtype, device=helpers.outputDevice())
        torch.randperm(n, out=res)
        return res

    randperm_harness(fn, n, dtype)
