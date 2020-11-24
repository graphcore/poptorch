#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import pytest

torch.manual_seed(42)
params_einsum = [
    ('ij->i', [torch.randn(5, 4)]),
    ('i,j->j', [torch.randn(5), torch.randn(4)]),
    ('i,j->ji', [torch.randn(5), torch.randn(4)]),
    ('bij,bjk->bik', [torch.randn(3, 2, 5),
                      torch.randn(3, 5, 4)]),
    ('bn,anm,bm->ba',
     [torch.randn(2, 5),
      torch.randn(3, 5, 4),
      torch.randn(2, 4)]),
    ('bfnd,ndh->bfh', [torch.randn(2, 3, 4, 5),
                       torch.randn(4, 5, 6)]),
    ('nmku,buvm->bnkv', [torch.randn(2, 3, 4, 5),
                         torch.randn(6, 5, 7, 3)]),
]


@pytest.mark.parametrize("params", params_einsum)
@pytest.mark.parametrize("implicit_rhs", {True, False})
def test_einsum(params, implicit_rhs):
    class Model(torch.nn.Module):
        def forward(self, xs):
            eq = params[0].split('->')[0] if implicit_rhs else params[0]
            return torch.einsum(eq, xs)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    # TODO(T26908): Passing in list causes compiler assert
    xs = tuple(params[1])

    # Run on CPU
    native_out = model(xs)

    # Run on IPU
    poptorch_out = poptorch_model(xs)

    assert native_out.size() == poptorch_out.size()
    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("arr_lengths",
                         ([3], [3, 3], [2, 4], [3, 2, 4], [5, 2, 3, 4]))
def test_meshgrid(arr_lengths):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, xs):
            return torch.meshgrid(xs)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    # TODO(T26908): Passing in list causes compiler assert
    xs = tuple([torch.randn(arr_length) for arr_length in arr_lengths])

    # Run on CPU
    native_out = model(xs)

    # Run on IPU
    poptorch_out = poptorch_model(xs)

    for native, pop in zip(native_out, poptorch_out):
        assert native.size() == pop.size()
        torch.testing.assert_allclose(native, pop)


@pytest.mark.parametrize("arr_lengths",
                         ([3], [3, 3], [2, 4], [3, 2, 4], [5, 2, 3, 4]))
def test_cartesian_prod(arr_lengths):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, xs):
            # Add one to force compute the first test case
            return torch.cartesian_prod(*xs) + 1

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    # TODO(T26908): Passing in list causes compiler assert
    xs = tuple([torch.randn(arr_length) for arr_length in arr_lengths])

    # Run on CPU
    native_out = model(xs)

    # Run on IPU
    poptorch_out = poptorch_model(xs)

    assert native_out.size() == poptorch_out.size()
    torch.testing.assert_allclose(native_out, poptorch_out)


@pytest.mark.parametrize("dims",
                         (0, 2, ([], []), ([2], [0]), ([2, 3], [0, 1])))
def test_tensordot(dims):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.tensordot(x, y, dims)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    x = torch.randn(2, 3, 5, 4)
    y = torch.randn(5, 4, 1)

    # Run on CPU
    native_out = model(x, y)

    # Run on IPU
    poptorch_out = poptorch_model(x, y)

    torch.testing.assert_allclose(native_out, poptorch_out)
