#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest

import poptorch
import helpers

torch.manual_seed(42)
params_einsum = [
    ('i->', (torch.randn(5), )),
    ('ij->i', (torch.randn(5, 4), )),
    ('i,j->j', (torch.randn(5), torch.randn(4))),
    ('i,j->ji', (torch.randn(5), torch.randn(4))),
    ('bij,bjk->bik', (torch.randn(3, 2, 5), torch.randn(3, 5, 4))),
    ('bn,anm,bm->ba', (torch.randn(2, 5), torch.randn(3, 5,
                                                      4), torch.randn(2, 4))),
    ('bfnd,ndh->bfh', (torch.randn(2, 3, 4, 5), torch.randn(4, 5, 6))),
    ('nmku,buvm->bnkv', (torch.randn(2, 3, 4, 5), torch.randn(6, 5, 7, 3))),
]


@pytest.mark.parametrize("params", params_einsum)
@pytest.mark.parametrize("implicit_rhs", {True, False})
def test_einsum(params, implicit_rhs):

    eq = params[0].split('->')[0] if implicit_rhs else params[0]

    op = lambda *xs: torch.einsum(eq, *xs)
    xs = params[1]
    model = helpers.ModelWithWeights(op, xs[0].shape)
    poptorch_model = poptorch.trainingModel(model)

    # Run on CPU
    native_out, _ = model(xs)

    # Run on IPU
    poptorch_out, _ = poptorch_model(xs)

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


def test_einsum_chained():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            r = torch.einsum('b u k m, b u v m -> b k v', x, y)
            return torch.einsum('b h k n, b k v -> b h v n', z, r)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    x = torch.randn(1, 4, 16, 4, dtype=torch.float)
    y = torch.randn(1, 4, 16, 4, dtype=torch.float)
    z = torch.randn(1, 4, 16, 4, dtype=torch.float)

    native_out = model(x, y, z)
    poptorch_out = poptorch_model(x, y, z)

    helpers.assert_allclose(expected=native_out,
                            actual=poptorch_out,
                            rtol=1e-3,
                            atol=1e-3)


@pytest.mark.parametrize("arr_lengths",
                         ([3], [3, 3], [2, 4], [3, 2, 4], [5, 2, 3, 4]))
def test_meshgrid(arr_lengths):
    torch.manual_seed(42)

    xs = tuple([torch.randn(arr_length) for arr_length in arr_lengths])

    model = helpers.ModelWithWeights(torch.meshgrid,
                                     xs[0].shape,
                                     out_fn=lambda x: x[0])
    poptorch_model = poptorch.trainingModel(model)

    # Run on CPU
    native_out, _ = model(xs)

    # Run on IPU
    poptorch_out, _ = poptorch_model(xs)

    # Inference test - check outputs
    for native, pop in zip(native_out, poptorch_out):
        helpers.assert_allclose(expected=native, actual=pop)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("arr_lengths",
                         ([3], [3, 3], [2, 4], [3, 2, 4], [5, 2, 3, 4]))
def test_cartesian_prod(arr_lengths):
    torch.manual_seed(42)

    xs = tuple([torch.randn(arr_length) for arr_length in arr_lengths])

    model = helpers.ModelWithWeights(torch.cartesian_prod, xs[0].shape)
    poptorch_model = poptorch.trainingModel(model)

    # Run on CPU
    native_out, _ = model(xs)

    # Run on IPU
    poptorch_out, _ = poptorch_model(xs)

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("dims",
                         (0, 2, ([], []), ([2], [0]), ([2, 3], [0, 1])))
def test_tensordot(dims):
    torch.manual_seed(42)

    op = lambda a, b: torch.tensordot(a, b, dims)

    x = torch.randn(2, 3, 5, 4)
    y = torch.randn(5, 4, 1)

    model = helpers.ModelWithWeights(op, x.shape)
    poptorch_model = poptorch.trainingModel(model)

    # Run on CPU
    native_out, _ = model((x, y))

    # Run on IPU
    poptorch_out, _ = poptorch_model((x, y))

    # Inference test - check outputs
    helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dim", range(-3, 3))
def test_scatter_add(inplace, dim):
    class Model(torch.nn.Module):
        def __init__(self, dim, dim_size):
            super().__init__()
            self.dim = dim
            self.dim_size = dim_size
            self.inplace = inplace

        def forward(self, src, index):
            sz = list(src.shape)
            sz[self.dim] = self.dim_size
            out = torch.ones(sz)

            if self.inplace:
                return out.scatter_add_(self.dim, index, src)

            return out.scatter_add(self.dim, index, src)

    torch.manual_seed(42)
    x = torch.randn(4, 8, 16)
    dim_size = x.shape[dim] // 2
    index = torch.randint_like(x, high=dim_size).long()
    model = helpers.ModelWithWeights(Model(dim, dim_size), x.shape)

    cpu_out, _ = model((x, index))
    pop_model = poptorch.trainingModel(model)
    ipu_out, _ = pop_model((x, index))

    # Inference test - check outputs
    helpers.assert_allclose(actual=ipu_out, expected=cpu_out)

    # Training test - check weights changed
    pop_model.assert_weights_changed()
