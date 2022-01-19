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


def op_harness(op, *inputs, assert_fn=None, out_fn=None):
    model = helpers.ModelWithWeights(op, inputs[0].shape, out_fn=out_fn)
    poptorch_model = poptorch.trainingModel(model)

    # Run on CPU
    native_out, _ = model(inputs)

    # Run on IPU
    poptorch_out, _ = poptorch_model(inputs)

    if assert_fn is None:

        def assert_fn(native_out, poptorch_out):
            if isinstance(native_out, tuple):
                for native, pop in zip(native_out, poptorch_out):
                    helpers.assert_allclose(expected=native, actual=pop)
            else:
                helpers.assert_allclose(expected=native_out,
                                        actual=poptorch_out)

    # Inference test - check outputs
    assert_fn(native_out, poptorch_out)

    # Training test - check weights changed
    poptorch_model.assert_weights_changed()


@pytest.mark.parametrize("params", params_einsum)
@pytest.mark.parametrize("implicit_rhs", {True, False})
def test_einsum(params, implicit_rhs):

    eq = params[0].split('->')[0] if implicit_rhs else params[0]

    op = lambda *xs: torch.einsum(eq, *xs)
    op_harness(op, *params[1])


def test_einsum_chained():
    torch.manual_seed(42)

    def op(x, y, z):
        r = torch.einsum('b u k m, b u v m -> b k v', x, y)
        return torch.einsum('b h k n, b k v -> b h v n', z, r)

    inputs = [torch.randn(1, 4, 16, 4, dtype=torch.float) for _ in range(3)]

    def assert_fn(native_out, poptorch_out):
        helpers.assert_allclose(expected=native_out,
                                actual=poptorch_out,
                                rtol=1e-3,
                                atol=1e-3)

    op_harness(op, *inputs, assert_fn=assert_fn)


@pytest.mark.parametrize("arr_lengths",
                         ([3], [3, 3], [2, 4], [3, 2, 4], [5, 2, 3, 4]))
def test_meshgrid(arr_lengths):
    torch.manual_seed(42)

    inputs = [torch.randn(arr_length) for arr_length in arr_lengths]

    op_harness(torch.meshgrid, *inputs, out_fn=lambda x: x[0])


@pytest.mark.parametrize("arr_lengths",
                         ([3], [3, 3], [2, 4], [3, 2, 4], [5, 2, 3, 4]))
def test_cartesian_prod(arr_lengths):
    torch.manual_seed(42)

    inputs = [torch.randn(arr_length) for arr_length in arr_lengths]

    op_harness(torch.cartesian_prod, *inputs)


@pytest.mark.parametrize("dims", (2, ([2], [0]), ([2, 3], [0, 1])))
def test_tensordot(dims):
    torch.manual_seed(42)

    op = lambda a, b: torch.tensordot(a, b, dims)

    x = torch.randn(2, 3, 5, 4)
    y = torch.randn(5, 4, 1)

    op_harness(op, x, y)


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

    op_harness(Model(dim, dim_size), x, index)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_available_memory_scatter_add(capfd):
    class Model(torch.nn.Module):
        def __init__(self, dim, dim_size):
            super().__init__()
            self.dim = dim
            self.dim_size = dim_size

        def forward(self, src, index):
            sz = list(src.shape)
            sz[self.dim] = self.dim_size
            out = torch.ones(sz)
            sa = out.scatter_add(self.dim, index, src)
            am = poptorch.set_available_memory(sa, 0.9)
            return am

    dim = 2
    torch.manual_seed(42)
    x = torch.randn(4, 8, 16)
    dim_size = x.shape[dim] // 2
    index = torch.randint_like(x, high=dim_size).long()

    model = Model(dim, dim_size)
    options = poptorch.Options()
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_model(x, index)

    log = helpers.LogChecker(capfd)
    it = log.createIterator()
    # Assert that the set_available_memory node references the scatterreduce,
    # not the add.
    sa_line = it.findNext("popart::scatterreduce").strip()
    sa_var = sa_line.partition(" ")[0]
    sam_line = it.findNext("poptorch::set_available_memory").strip()
    assert sam_line.endswith("({})".format(sa_var))
