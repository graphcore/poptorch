#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import re

import torch
import pytest
import helpers
import poptorch

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


def test_einsum_transpose():
    torch.manual_seed(42)

    def op(x):
        return torch.einsum('n c h w -> n h w c', x)

    inputs = [torch.randn(2, 3, 4, 5, dtype=torch.float)]

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


@pytest.mark.parametrize("dim", range(-3, 3))
@pytest.mark.parametrize("reduce", ['sum', 'amin', 'amax', 'mean'])
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce(dim, reduce, include_self):

    if reduce in ['amin', 'amax']:
        pytest.skip("Known issue. Unblock when AFS-78 will be completed.")

    class Model(torch.nn.Module):
        def __init__(self, dim, reduce, include_self):
            super().__init__()
            self.dim = dim
            self.reduce = reduce
            self.include_self = include_self

        def forward(self, inp, index, src):
            output = inp.scatter_reduce(self.dim,
                                        index,
                                        src,
                                        reduce=self.reduce,
                                        include_self=self.include_self)
            return output

    torch.manual_seed(42)
    src = torch.randn(4, 8, 16)
    dim_size = src.shape[dim] // 2
    sz = list(src.shape)
    sz[dim] = dim_size
    inp = torch.randn(sz)
    index = torch.randint_like(src, high=dim_size).long()
    op_harness(Model(dim, reduce, include_self), inp, index, src)


@pytest.mark.parametrize("dim", range(-3, 3))
@pytest.mark.parametrize("reduce", ["mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce(dim, reduce, include_self):

    if reduce in ['amin', 'amax']:
        pytest.skip("Known issue. Unblock when AFS-78 will be completed.")

    class Model(torch.nn.Module):
        def __init__(self, dim, reduce, include_self):
            super().__init__()
            self.dim = dim
            self.reduce = reduce
            self.include_self = include_self

        def forward(self, inp, index, src):
            output = inp.index_reduce_(self.dim,
                                       index,
                                       src,
                                       reduce=self.reduce,
                                       include_self=self.include_self)
            return output

    torch.manual_seed(17)
    inp = torch.randn(5, 8, 11)
    dim_size = inp.shape[dim] // 2
    sz = list(inp.shape)
    sz[dim] = dim_size
    src = torch.randn(sz)
    index = torch.randint(high=dim_size, size=(dim_size, )).long()
    op_harness(Model(dim, reduce, include_self), inp, index, src)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.parametrize("reduce", ['sum', 'amin', 'amax', 'mean'])
@pytest.mark.parametrize("expand_as", [True, False])
@pytest.mark.parametrize("include_self", [True, False])
def test_2d_scatter_reduce_with_index_expansion(capfd, reduce, expand_as,
                                                include_self):
    if reduce in ['amin', 'amax']:
        pytest.skip("Known issue. Unblock when AFS-78 will be completed.")

    class Model(torch.nn.Module):
        def __init__(self, reduce, include_self):
            super().__init__()
            self.reduce = reduce
            self.include_self = include_self

        def forward(self, inp, index, src):
            if expand_as:
                index = index.expand_as(src)
            else:
                index = index.expand(src.shape)
            output = inp.scatter_reduce(-2,
                                        index,
                                        src,
                                        reduce=self.reduce,
                                        include_self=self.include_self)
            return output

    model = Model(reduce, include_self)
    poptorch_model = poptorch.inferenceModel(model)

    torch.manual_seed(0)
    index = torch.randint(0, 5, (6, 1), dtype=torch.long)
    src = torch.rand((6, 3))
    inp = torch.randn((5, 3))
    out = model(inp, index, src)
    poptorch_out = poptorch_model(inp, index, src)
    helpers.assert_allclose(actual=poptorch_out, expected=out)

    # Make sure the expand op is removed.
    look_for = "aten::expand_as" if expand_as else "aten::expand"
    log = helpers.LogChecker(capfd)
    it = log.createIterator()
    it.findNext("Removing index expansion node:")
    it.assert_not_contains(look_for)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.parametrize("expand_as", [True, False])
def test_2d_scatter_add_with_index_expansion(capfd, expand_as):
    class Model(torch.nn.Module):
        def forward(self, index, src):
            if expand_as:
                index = index.expand_as(src)
            else:
                index = index.expand(src.shape)
            return torch.zeros((5, 3)).scatter_add_(
                dim=-2,
                index=index,
                src=src,
            )

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    torch.manual_seed(0)
    index = torch.randint(0, 5, (6, 1), dtype=torch.long)
    src = torch.rand((6, 3))

    out = model(index, src)
    poptorch_out = poptorch_model(index, src)
    helpers.assert_allclose(actual=poptorch_out, expected=out)

    # Make sure the expand op is removed.
    look_for = "aten::expand_as" if expand_as else "aten::expand"
    log = helpers.LogChecker(capfd)
    it = log.createIterator()
    it.findNext("Removing index expansion node:")
    it.assert_not_contains(look_for)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.parametrize("expand_as", [True, False])
@pytest.mark.parametrize("params", [
    {
        "shape": (3, 5),
        "gather_dim": 0,
        "expand_dim": 0,
        "should_optimise": False
    },
    {
        "shape": (3, 5),
        "gather_dim": 0,
        "expand_dim": 1,
        "should_optimise": True
    },
    {
        "shape": (3, 5),
        "gather_dim": 1,
        "expand_dim": 0,
        "should_optimise": True
    },
    {
        "shape": (3, 5),
        "gather_dim": 1,
        "expand_dim": 1,
        "should_optimise": False
    },
    {
        "shape": (1, 1, 3, 1, 5, 1),
        "gather_dim": 3,
        "expand_dim": 2,
        "should_optimise": False
    },
    {
        "shape": (1, 1, 3, 1, 5, 1),
        "gather_dim": 2,
        "expand_dim": 4,
        "should_optimise": True
    },
    {
        "shape": (1, 1, 3, 1, 5, 1),
        "gather_dim": 4,
        "expand_dim": 2,
        "should_optimise": True
    },
    {
        "shape": (1, 1, 3, 1, 5, 1),
        "gather_dim": 4,
        "expand_dim": 1,
        "should_optimise": False
    },
    {
        "shape": (3, 4, 5),
        "gather_dim": 0,
        "expand_dim": 1,
        "should_optimise": False
    },
])
def test_gather_with_index_expansion(capfd, expand_as, params):
    # Work out params to model.
    torch.manual_seed(42)

    data = torch.randint(10, params["shape"], dtype=torch.int)

    indices_shape = list(data.shape)
    indices_shape[params["expand_dim"]] = 1
    indices = torch.randint(high=data.shape[params["gather_dim"]],
                            size=indices_shape)

    # Make model.
    class Model(torch.nn.Module):
        def forward(self, data, indices):
            if expand_as:
                indices = indices.expand_as(data)
            else:
                indices = indices.expand(data.shape)

            # Also do an `add`, to check we can pipe the results onward.
            return torch.gather(data, params["gather_dim"], indices).add(8)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    # Run model, check result is still correct.
    cpu_out = model(data, indices)
    ipu_out = poptorch_model(data, indices)
    helpers.assert_allclose(actual=ipu_out, expected=cpu_out)

    log = helpers.LogChecker(capfd)
    it = log.createIterator()

    # Look for the log saying we did the optimisation, only if we should have.
    if params["should_optimise"]:
        it.findNext("Optimising gather:")

    # Look for the (non-)presence of the expand op that should be removed.
    # Note: aten::expand_as might be intercepted as aten::expand by the dispatcher
    # so only check for "expand"
    remove_if_optimised = "aten::expand"

    if params["should_optimise"]:
        it.assert_not_contains(remove_if_optimised)
    else:
        it.findNext(remove_if_optimised)


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
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x, index)

    log = helpers.LogChecker(capfd)
    it = log.createIterator()
    it.findNext("Graph before lowering to PopART:")
    # Assert that the set_available_memory node references the scatterreduce,
    # not the add.
    sa_line = it.findNext("popart::scatterreduce").strip()
    sa_var = sa_line.partition(" ")[0]
    sam_line = it.findNext("poptorch::set_available_memory").strip()
    # Check we have set_available_memory[...](%XX) where XX is the result of scatterreduce
    assert re.search(r"set_available_memory\[.*\]\(\{}\)".format(sa_var),
                     sam_line)
