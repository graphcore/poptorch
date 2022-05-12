#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import pytest
import helpers
from poptorch.experimental import IPUContext


# Helper, that takes a PyTorchy-function to run on the CPU & IPU.
#
# *args & **kwargs are forwarded straight to the given function.
#
# results are compared with the cpu version with assert_allclose
def reduce_harness(fn, *args, **kwargs):
    # Get the result from the CPU
    cpu_res = fn(*args, **kwargs)
    print(f"From CPU: {cpu_res}")

    ipu_res = IPUContext(fn)(*args, **kwargs)
    print(f"From IPU: {ipu_res}")

    # TODO(T62262): Handle int64s properly
    check_dtype = isinstance(
        cpu_res, torch.Tensor
    ) and cpu_res.dtype != torch.int64 and cpu_res.dtype != torch.int32
    return helpers.assert_allclose(expected=cpu_res,
                                   actual=ipu_res,
                                   check_dtype=check_dtype,
                                   equal_nan=True)


# This seed has a failure when computing standard deviation due to numerical stability issues
# Note: the seed is used in the float test cases for a large (ish) tensor
torch.manual_seed(11219477931810481670)

int_test_cases = [
    torch.tensor(3),
    torch.tensor([0, 1, 2, 3]),
    torch.tensor([[0, 1], [2, 3]])
]
uint8_test_cases = [
    torch.tensor(3, dtype=torch.uint8),
    torch.tensor([], dtype=torch.uint8),
    torch.tensor([0, 0], dtype=torch.uint8),
    torch.tensor([[1, 4], [4, 5]], dtype=torch.uint8),
]
bool_test_cases = [
    torch.tensor(False),
    torch.tensor([True, False]),
    torch.tensor([True, True]),
    torch.tensor([[False, False], [False, False]])
]
float_test_cases = [
    torch.tensor(3.),
    torch.tensor([]),
    # Similar values with large mean to demonstrate numerical stability issues
    # in standard deviation
    torch.tensor([10.9123, 10.9108, 10.9119]),
    torch.tensor([[0., 1.], [2., 3.]]),
    torch.rand(3, 32, 128),
    torch.empty(0, 3, 5),
    torch.empty(1, 0, 5),
    torch.empty(1, 3, 0)
]

all_test_cases = (int_test_cases + uint8_test_cases + bool_test_cases +
                  float_test_cases)


# TODO(T62262): Test with int64 dtypes
@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("func", [torch.sum, torch.prod])
@pytest.mark.parametrize("input", all_test_cases)
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.int32])
def test_sum_prod(func, input, dtype):
    reduce_harness(func, input, dtype=dtype)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("func", [torch.sum, torch.prod])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("input", all_test_cases)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.int32])
def test_sum_prod_dim(func, input, dim, keepdim, dtype):
    reduce_harness(func, input, dim, keepdim, dtype=dtype)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input", float_test_cases)
@pytest.mark.parametrize("dtype", [None, torch.float16, torch.float32])
def test_mean(input, dtype):
    reduce_harness(torch.mean, input, dtype=dtype)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input", float_test_cases)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [None, torch.float16, torch.float32])
def test_mean_dim(input, dim, keepdim, dtype):
    reduce_harness(torch.mean, input, dim, keepdim, dtype=dtype)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize(
    "func", [torch.std, torch.var, torch.std_mean, torch.var_mean])
@pytest.mark.parametrize("input", float_test_cases)
@pytest.mark.parametrize("unbiased", [True, False])
def test_stats(func, input, unbiased):
    reduce_harness(func, input, unbiased)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize(
    "func", [torch.std, torch.var, torch.std_mean, torch.var_mean])
@pytest.mark.parametrize("input", float_test_cases)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("keepdim", [True, False])
def test_stats_dim(func, input, dim, unbiased, keepdim):
    reduce_harness(func, input, dim, unbiased, keepdim)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("func", [torch.all, torch.any])
@pytest.mark.parametrize("input", all_test_cases)
def test_any_all(func, input):
    reduce_harness(func, input)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("func", [torch.all, torch.any])
@pytest.mark.parametrize("input", all_test_cases)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_any_all_dim(func, input, dim, keepdim):
    reduce_harness(func, input, dim, keepdim)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("input", all_test_cases)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.int32])
def test_cumsum(input, dim, dtype):
    reduce_harness(torch.cumsum, input, dim, dtype=dtype)
