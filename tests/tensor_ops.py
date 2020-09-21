#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch

import pytest

# Tensors

# Creation ops (we don't support many of these)
# torch.numel, torch.tensor, torch.sparse_coo_tensor, torch.as_tensor, torch.as_strided, torch.from_numpy, torch.zeros,
# torch.zeros_like, torch.ones, torch.ones_like, torch.arange, torch.range, torch.linspace, torch.logspace, torch.eye,
# torch.empty, torch.empty_like, torch.empty_strided, torch.full, torch.full_like, torch.quantize_per_tensor, torch.quantize_per_channel,

# Indexing, Slicing, Joining, Mutating Ops
# torch.cat, torch.chunk, torch.gather, torch.index_select, torch.masked_select, torch.narrow, torch.nonzero, torch.reshape, torch.split,
# torch.squeeze, torch.stack, torch.t, torch.take, torch.transpose, torch.unbind, torch.unsqueeze, torch.where, torch._C.Generator,
# torch._C.Generator.device,


def test_zeros_and_ones():
    class Model(torch.nn.Module):
        def forward(self):
            x = torch.zeros(3, 5, 1)
            y = torch.ones(3, 5, 1)

            # A stupid test to stop popart from prunning this.
            return x * y, y + x

    model = Model()

    # Run on CPU.
    nativeOut = model()

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model()

    assert torch.equal(nativeOut[0], poptorch_out[0])
    assert torch.equal(nativeOut[1], poptorch_out[1])


def test_cat():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cat((x, x, x), 0)

    model = Model()
    x = torch.randn(2, 3)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_chunk():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.chunk(x, 5)

    model = Model()
    x = torch.randn(20, 10)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    for native, pop in zip(nativeOut, poptorch_out):
        assert torch.equal(native, pop)


def test_reshape():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.reshape(x, (1, 1, 2, 2))

    model = Model()
    x = torch.arange(4.)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_split():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.split(x, 5)

    model = Model()
    x = torch.randn(20, 10)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    for native, pop in zip(nativeOut, poptorch_out):
        assert torch.equal(native, pop)


def test_squeeze():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.squeeze(x)

    model = Model()
    x = torch.randn(1, 1, 20, 1, 10, 1)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_t():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.t(x)

    model = Model()
    x = torch.randn(20, 10)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_transpose():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.transpose(x, 3, 0)

    model = Model()
    x = torch.randn(3, 2, 5, 10)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_unsqueeze():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.unsqueeze(x, 1)

    model = Model()
    x = torch.randn(3, 2, 5, 10)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_expand():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.expand(3, 4)

    model = Model()
    x = torch.randn(3, 1)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_expand_as():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x.expand_as(y)

    model = Model()
    x = torch.randn(3, 1)
    y = torch.randn(3, 4)

    # Run on CPU.
    nativeOut = model(x, y)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x, y)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_flatten():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.flatten(x)

    model = Model()
    x = torch.randn(3, 1)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


def test_view():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.view((15, 2, 5))

    model = Model()
    x = torch.randn(30, 5)

    # Run on CPU.
    nativeOut = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert nativeOut.size() == poptorch_out.size()
    assert torch.equal(nativeOut, poptorch_out)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 2), (2, 3, 4)])
def test_size(input_shapes):
    class Model(torch.nn.Module):
        def forward(self, x):
            # Use size as input to another operation to workaround pruning error
            return x.view(x.size())

    model = Model()
    x = torch.ones(*input_shapes)

    # Run on CPU.
    native_out = model(x)
    assert torch.equal(x, native_out)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    assert native_out.size() == poptorch_out.size()
    assert torch.equal(native_out, poptorch_out)


input_shapes = [(1, 4, 5), (2, ), (2, 2), (2, 3, 4, 1, 3, 4)]
dtypes = [torch.float, torch.float16, torch.int32]


@pytest.mark.parametrize("input_shapes", input_shapes)
@pytest.mark.parametrize("t", dtypes)
def test_fill(input_shapes, t):
    float_test_num = 1.9375

    class Model(torch.nn.Module):
        def forward(self, x):
            value = 42 if x.dtype == torch.int32 else float_test_num
            return x.fill_(value), torch.full_like(x, value), torch.full(
                input_shapes, value, dtype=x.dtype)

    model = Model()
    x = torch.ones(*input_shapes, dtype=t)

    # Run on CPU.
    if t != torch.float16:
        native_out = model(x)
    else:
        native_out = (torch.full(input_shapes, float_test_num),
                      torch.full(input_shapes, float_test_num),
                      torch.full(input_shapes, float_test_num))

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    for native, pop in zip(native_out, poptorch_out):
        if t == torch.float16:
            pop = pop.float()

        assert native.size() == pop.size()
        assert torch.equal(native, pop)
        assert native.dtype == pop.dtype


@pytest.mark.parametrize("input_shapes", input_shapes)
@pytest.mark.parametrize("value", [0.666, -4.32, float("Inf"), float("-Inf")])
def test_masked_fill(input_shapes, value):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x):
            fill_result = x.masked_fill(x > 0.5, value)
            where_result = torch.where(x > 0.5, x, torch.tensor(value))
            return fill_result, where_result

    model = Model()
    x = torch.randn(*input_shapes)

    # Run on CPU.
    native_out = model(x)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x)

    for pop, native in zip(poptorch_out, native_out):
        assert native.size() == pop.size()
        assert torch.equal(native, pop)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (3, 4), (1, 3, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_stack(input_shapes, dim):

    if dim > len(input_shapes):
        pytest.skip()

    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            return torch.stack([x, y, z], dim=dim)

    model = Model()
    a = torch.randn(*input_shapes)
    b = torch.randn(*input_shapes)
    c = torch.randn(*input_shapes)

    # Run on CPU.
    native_out = model(a, b, c)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(a, b, c)

    for pop, native in zip(poptorch_out, native_out):
        assert native.size() == pop.size()
        assert torch.equal(native, pop)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dims",
                         [[1], [3], [2, 1], [2, 3], [1, 1, 1], [3, 2, 4]])
def test_repeat(input_shapes, dims):

    if len(dims) < len(input_shapes):
        pytest.skip(
            "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor."
        )

    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x):
            return x.repeat(dims)

    model = Model()
    a = torch.randn(*input_shapes)

    # Run on CPU.
    native_out = model(a)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(a)

    for pop, native in zip(poptorch_out, native_out):
        assert native.size() == pop.size()
        assert torch.equal(native, pop)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.int])
def test_copy_(input_shapes, dtype):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return y.copy_(x)

    model = Model()
    x = torch.randn(*input_shapes)
    y = torch.empty_like(x, dtype=dtype)

    # Run on CPU.
    native_out = model(x, y)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(x, y)

    for pop, native in zip(poptorch_out, native_out):
        assert native.size() == pop.size()
        assert native.dtype == pop.dtype
        assert torch.equal(native, pop)
