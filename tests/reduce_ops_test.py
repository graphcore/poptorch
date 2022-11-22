#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import torch
import pytest
import numpy as np
import helpers
import poptorch


# Reduce Ops Harness
# Checks that the IPU reduce ops match the CPU version.
def reduce_harness(func, input, **kwargs):
    # pass any reduce op kwargs only if they're set to
    # avoid named tensor errors
    op_kwargs = {name: val for name, val in kwargs.items() if val is not None}

    def reduce_op(x):
        return func(x, **op_kwargs)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reduce_op = reduce_op

        def forward(self, x):
            # Ensure input is not modified in place
            x = x + 0
            return self.reduce_op(x)

    model = Model()

    # Run on IPU and check that the result has the correct type
    pop_model = poptorch.inferenceModel(model)
    pop_out = pop_model(input)
    native_out = model(input)

    check_dtype = "dtype" in kwargs
    if torch.is_floating_point(native_out):
        helpers.assert_allclose(expected=native_out,
                                actual=pop_out,
                                check_dtype=check_dtype)
    else:
        helpers.assert_allequal(expected=native_out,
                                actual=pop_out,
                                check_dtype=check_dtype)


# torch.all, torch.any
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("func", [torch.all, torch.any])
def test_any_all(func, dim):
    input = torch.randint(low=0, high=3, size=(32, 128))
    reduce_harness(func, input, dim=dim)


@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("func", [torch.sum, torch.mean])
def test_sum_mean(func, dim):
    input = torch.rand(32, 128)
    reduce_harness(func, input, dim=dim)


@pytest.mark.parametrize("dim", (None, 0, -1, [1, 2]))
def test_count_nonzero(dim):
    torch.manual_seed(42)
    input = torch.randint(10, (2, 3, 4, 5))
    reduce_harness(torch.count_nonzero, input, dim=dim)


@pytest.mark.parametrize("dim", (None, 0, -1, [0, 1]))
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_nansum(dim, keepdim, dtype):
    np.random.seed(0)
    # Create a tensor that contains some nans - be careful in the
    # torch.float16 case to not overflow the float16 range
    shape = (10, 10)
    mask = np.random.randint(0, 2, size=shape).astype(bool)
    data = np.random.rand(*shape).astype(np.float32)
    data[mask] = np.nan

    input = torch.from_numpy(data)
    reduce_harness(
                   torch.nansum,
                   input,
                   dim=dim,
                   keepdim=keepdim,
                   dtype=dtype)
