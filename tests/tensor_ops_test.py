#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import copy
import pytest
import torch
import helpers
import poptorch

# Tensors

# Creation ops (we don't support many of these)
# torch.numel, torch.tensor, torch.sparse_coo_tensor, torch.as_tensor, torch.as_strided, torch.from_numpy, torch.zeros,
# torch.zeros_like, torch.ones, torch.ones_like, torch.arange, torch.range, torch.linspace, torch.logspace, torch.eye,
# torch.empty, torch.empty_like, torch.empty_strided, torch.full, torch.full_like, torch.quantize_per_tensor, torch.quantize_per_channel,

# Indexing, Slicing, Joining, Mutating Ops
# torch.cat, torch.chunk, torch.gather, torch.index_select, torch.masked_select, torch.narrow, torch.nonzero, torch.reshape, torch.split,
# torch.squeeze, torch.stack, torch.t, torch.take, torch.transpose, torch.unbind, torch.unsqueeze, torch.where, torch._C.Generator,
# torch._C.Generator.device,


def zeros_and_ones_harness(model, dtype, is_like):
    assert dtype in [torch.float16, torch.float32, torch.int32, torch.bool]
    torch.manual_seed(42)

    # Calculating with ints/bools does not produce meaningful gradients
    test_training = not dtype in (torch.int32, torch.bool)

    inputs = [torch.tensor([1], dtype=dtype)]
    if is_like:
        inputs.append(torch.empty(3, 5, 1))
    inputs = tuple(inputs)

    if test_training:
        out_fn = lambda out: out[0]
        model = helpers.ModelWithWeights(model, inputs[0].shape, out_fn=out_fn)
        # We need to copy the model to use the original weights for native comparison
        model_copy = copy.deepcopy(model)
        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model)
        poptorch_out, _ = poptorch_model(inputs)
        if dtype is torch.float16:
            # Promote CPU model and input
            model_copy = model_copy.float()
            inputs = tuple([input.float() for input in inputs])
            # promote IPU result to allow comparison
            poptorch_out = [pop.float() for pop in poptorch_out]
        native_out, _ = model_copy(inputs)
    else:
        native_out = model(*inputs)
        poptorch_model = poptorch.inferenceModel(model)
        poptorch_out = poptorch_model(*inputs)

    # Inference test - check outputs
    for native, pop in zip(native_out, poptorch_out):
        rtol = 0.001 if dtype is torch.float16 else 0.0001
        atol = 1e-4 if dtype is torch.float16 else 1e-5
        helpers.assert_allclose(expected=native,
                                actual=pop,
                                rtol=rtol,
                                atol=atol)

    if test_training:
        # Training test - check weights changed
        poptorch_model.assert_weights_changed()


zeros_and_ones_dtypes = [torch.float16, torch.float32, torch.int32, torch.bool]


@pytest.mark.parametrize("dtype", zeros_and_ones_dtypes)
def test_zeros_and_ones(dtype):
    class Model(torch.nn.Module):
        def forward(self, z):
            x = torch.zeros(3, 5, 1, dtype=dtype)
            y = torch.ones(3, 5, 1, dtype=dtype)

            return (x * y) + z, (y + x) + z

    zeros_and_ones_harness(Model(), dtype, False)


@pytest.mark.parametrize("dtype", zeros_and_ones_dtypes)
def test_new_zeros_and_new_ones(dtype):
    class Model(torch.nn.Module):
        def forward(self, z):
            x = z.new_zeros(3, 5, 1)
            y = z.new_ones(3, 5, 1)

            return (x * y) + z, (y + x) + z

    zeros_and_ones_harness(Model(), dtype, False)


@pytest.mark.parametrize("dtype", zeros_and_ones_dtypes)
def test_zeros_like_and_ones_like(dtype):
    class Model(torch.nn.Module):
        def forward(self, z, t):
            x = torch.zeros_like(t, dtype=dtype)
            y = torch.ones_like(t, dtype=dtype)

            return (x * y) + z, (y + x) + z

    zeros_and_ones_harness(Model(), dtype, True)


def op_harness(op,
               *inputs,
               test_training=True,
               assert_fn=None,
               out_fn=None,
               native_out=None):

    if assert_fn is None:

        def assert_fn(native_out, poptorch_out):
            if isinstance(native_out, tuple):
                for native, pop in zip(native_out, poptorch_out):
                    helpers.assert_allclose(expected=native, actual=pop)
            else:
                helpers.assert_allclose(expected=native_out,
                                        actual=poptorch_out)

    if test_training:
        # Set a fixed seed for the weights of the model
        torch.manual_seed(42)
        model = helpers.ModelWithWeights(op, inputs[0].shape, out_fn=out_fn)

        # Run on CPU.
        if native_out is None:
            native_out, _ = model(inputs)

            # native_out could be an alias of the input and so modified by
            # the poptorch_model
            if isinstance(native_out, tuple):
                native_out = tuple([n.clone().detach() for n in native_out])
            else:
                native_out = native_out.clone().detach()

        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model)
        poptorch_out, _ = poptorch_model(inputs)

        # Training test - check weights changed
        poptorch_model.assert_weights_changed()
    else:
        model = torch.nn.Module()
        model.forward = op

        # Run on CPU.
        if native_out is None:
            native_out = model(*inputs)

        poptorch_model = poptorch.inferenceModel(model)
        # Run on IPU.
        poptorch_out = poptorch_model(*inputs)

    # Inference test - check outputs
    assert_fn(native_out, poptorch_out)


# Note: Many of the following operations don't depend on the values of the tensors
# but we still need to fix the random seed for any op with randomly generated values
# so that it's guaranteed that weights change after one training step


@pytest.mark.parametrize("dim", [0, 1])
def test_cat(dim):
    torch.manual_seed(42)
    x = torch.randn(2, 3)

    op = lambda *xs: torch.cat(xs, dim=dim)

    op_harness(op, x, x, x)


def test_chunk():
    torch.manual_seed(42)
    x = torch.randn(20, 10)

    op = lambda x: torch.chunk(x, 5)

    op_harness(op, x, out_fn=lambda x: x[0])


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2])
def test_gather_3dim(dim):
    torch.manual_seed(42)
    shape = (9, 11, 6)
    input = torch.randn(shape)

    indices = torch.randint(0, 6, shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices)

    small_shape = (7, 9, 5)
    indices = torch.randint(0, 6, small_shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_gather_4dim(dim):
    torch.manual_seed(42)
    shape = (5, 8, 6, 7)
    input = torch.randn(shape)

    indices = torch.randint(0, 5, shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices)

    small_shape = (4, 5, 2, 6)
    indices = torch.randint(0, 5, small_shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices)


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_gather_5dim(dim):
    torch.manual_seed(42)
    shape = (3, 3, 3, 3, 3)
    input = torch.randn(shape)

    indices = torch.randint(0, 3, shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices)

    small_shape = (2, 2, 2, 2, 2)
    indices = torch.randint(0, 3, small_shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices)


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_scatter(dim):
    shape = (3, 3, 3, 3, 3)
    input = torch.randn(shape)
    source = torch.randn(shape)

    indices = torch.randint(0, 3, shape)
    op = lambda inp, idx, src: inp.scatter_(dim, idx, source)
    op_harness(op, input, indices, source)


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_scatter_(dim):
    shape = (3, 3, 3, 3, 3)
    input = torch.randn(shape)
    source = torch.randn(shape)

    indices = torch.randint(0, 3, shape)
    op = lambda inp, idx, src: inp.scatter_(dim, idx, source)
    op_harness(op, input, indices, source)


def test_reshape():
    op = lambda x: torch.reshape(x, (1, 1, 2, 2))

    x = torch.arange(4.)

    op_harness(op, x)


@pytest.mark.parametrize("split_size_or_sections",
                         (1, 5, 6, 20, [10, 10], [19, 1]))
def test_split(split_size_or_sections):
    torch.manual_seed(42)
    x = torch.randn(20, 10)
    op = lambda x: torch.split(x, split_size_or_sections)

    op_harness(op, x, out_fn=lambda x: x[0])


def test_split_singleton():
    torch.manual_seed(42)
    x = torch.randn(1, 4, 3, 1)
    op = lambda x: torch.split(x, 1, 1)[0]

    op_harness(op, x)


def test_squeeze():
    torch.manual_seed(42)
    x = torch.randn(1, 1, 5, 1, 10, 1)

    op_harness(torch.squeeze, x)


def test_t():
    torch.manual_seed(42)
    x = torch.randn(20, 10)

    op_harness(torch.t, x)


def test_transpose():
    torch.manual_seed(42)
    x = torch.randn(3, 2, 5, 2)
    op = lambda x: torch.transpose(x, 3, 0)

    op_harness(op, x)


def test_numpy_T():
    torch.manual_seed(42)
    op = lambda x: x.T

    x = torch.randn(3, 2, 5, 4)
    op_harness(op, x)

    x = torch.randn(5)
    op_harness(op, x)


def test_unsqueeze():
    torch.manual_seed(42)
    x = torch.randn(3, 2, 5, 2)
    op = lambda x: torch.unsqueeze(x, 1)

    op_harness(op, x)


def test_expand():
    torch.manual_seed(42)
    x = torch.randn(3, 1)
    op = lambda x: x.expand(3, 4)

    op_harness(op, x)


def test_expand_preserve_dim():
    torch.manual_seed(42)
    x = torch.randn(1, 1, 100)
    op = lambda x: x.expand(2, -1, -1)

    op_harness(op, x)


def test_expand_as():
    torch.manual_seed(42)
    x = torch.randn(3, 1)
    y = torch.randn(3, 4)
    op = lambda x, y: x.expand_as(y)

    op_harness(op, x, y)


def test_flatten():
    torch.manual_seed(42)
    x = torch.randn(3, 1)

    op_harness(torch.flatten, x)


def test_view():
    torch.manual_seed(42)
    x = torch.randn(30, 5)
    op = lambda x: x.view((15, 2, 5))

    op_harness(op, x)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 2), (2, 3, 4)])
def test_size(input_shapes):
    x = torch.ones(*input_shapes)
    # Use size as input to another operation to workaround pruning error
    op = lambda x: x.view(x.size())

    op_harness(op, x)


input_shapes = [(1, 4, 5), (2, ), (2, 2), (2, 3, 4, 1, 3, 4)]
dtypes = [torch.float, torch.float16, torch.int32]


@pytest.mark.parametrize("input_shapes", input_shapes)
@pytest.mark.parametrize("t", dtypes)
def test_fill(input_shapes, t):
    float_test_num = 1.9375

    def op(x):
        value = 42 if x.dtype == torch.int32 else float_test_num
        x = x + 0  # Ensure x is not modified in place
        return x.fill_(value), torch.full_like(x, value), torch.full(
            input_shapes, value, dtype=x.dtype)

    x = torch.ones(*input_shapes, dtype=t)

    native_out = tuple(
        [torch.full(input_shapes, float_test_num)
         for _ in range(3)]) if t == torch.float16 else None

    def assert_fn(native_out, poptorch_out):
        for native, pop in zip(native_out, poptorch_out):
            if t == torch.float16:
                pop = pop.float()

            assert native.dtype == pop.dtype
            helpers.assert_allequal(expected=native, actual=pop)

    # Fill is non-differentiable so set test_training=False
    op_harness(op,
               x,
               test_training=False,
               assert_fn=assert_fn,
               native_out=native_out)


@pytest.mark.parametrize("input_shapes", input_shapes)
@pytest.mark.parametrize("value", [0.666, -4.32, float("Inf"), float("-Inf")])
def test_masked_fill(input_shapes, value):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x):
            fill_result = x.masked_fill(x > 0.5, value)
            where_result = torch.where(x > 0.5, x, torch.tensor(value))
            return fill_result, where_result

    x = torch.randn(*input_shapes)
    op_harness(Model(), x, out_fn=lambda x: x[0])


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (3, 4), (1, 3, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_stack(input_shapes, dim):
    torch.manual_seed(42)

    if dim > len(input_shapes):
        pytest.skip()

    op = lambda *xs: torch.stack(xs, dim=dim)
    inputs = [torch.randn(*input_shapes) for _ in range(3)]

    op_harness(op, *inputs)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dims",
                         [[1], [3], [2, 1], [2, 3], [1, 1, 1], [3, 2, 4]])
def test_repeat(input_shapes, dims):

    if len(dims) < len(input_shapes):
        pytest.skip(
            "Number of dimensions of repeat dims can not be smaller than number"
            " of dimensions of tensor.")

    torch.manual_seed(42)

    op = lambda x: x.repeat(dims)
    a = torch.randn(*input_shapes)

    op_harness(op, a)


def test_repeat_training_input():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Dummy weights for training
            self.lin = torch.nn.Linear(2, 1)

        def forward(self, x):
            x = x.repeat(5, 2, 2)
            return x, poptorch.identity_loss(x**2, reduction="sum")

    torch.manual_seed(42)

    input = torch.randn((10, 1, 1))

    model = Model()
    poptorch_model = poptorch.trainingModel(model)

    native_out, _ = model(input)
    poptorch_out, _ = poptorch_model(input)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.int])
def test_clone_one(input_shapes, dtype):
    torch.manual_seed(42)

    op = lambda x: x.clone()

    x = torch.randn(*input_shapes)

    def assert_fn(native_out, poptorch_out):
        for pop, native in zip(poptorch_out, native_out):
            assert native.dtype == pop.dtype
            helpers.assert_allclose(expected=native, actual=pop)

    # Calculating with integers does not produce meaningful gradients
    test_training = dtype is torch.float
    op_harness(op, x, test_training=test_training, assert_fn=assert_fn)


def test_clone_two():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            x += y
            x_clone = x.clone()
            x += y
            x_clone += z

            return x, x_clone

    dummy_x = torch.randn([2, 3])
    dummy_y = torch.randn([2, 3])
    dummy_z = torch.randn([2, 3])

    model = Model()

    native_out = model(dummy_x.clone(), dummy_y.clone(), dummy_z.clone())

    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(dummy_x.clone(), dummy_y.clone(),
                                  dummy_z.clone())

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.half, torch.int])
def test_copy_(input_shapes, dtype):
    torch.manual_seed(42)

    op = lambda x, y: y.copy_(x)

    x = torch.randn(*input_shapes)
    y = torch.empty_like(x, dtype=dtype)

    def assert_fn(native_out, poptorch_out):
        for pop, native in zip(poptorch_out, native_out):
            assert native.dtype == pop.dtype
            helpers.assert_allclose(expected=native, actual=pop)

    # Calculating with integers does not produce meaningful gradients
    test_training = dtype is torch.float
    op_harness(op, x, y, test_training=test_training, assert_fn=assert_fn)


@pytest.mark.parametrize("shifts,dims", [(1, 0), (-1, 0), (10, 1), (-10, 1),
                                         (0, 2), ((1, 1), (0, 1)),
                                         ((1, -1), (1, 2)), ((-3, -4), (0, 2)),
                                         ((1, 2, 3), (0, 1, 2)),
                                         ((-1, -2, -3), (0, 1, 2)), (5, None),
                                         (-3, None)])
def test_roll(shifts, dims):
    torch.manual_seed(0)
    op = lambda x: x.roll(shifts, dims)
    x = torch.randn((2, 3, 4))
    op_harness(op, x)


@pytest.mark.parametrize("dims", [(0, 1)])
def test_flip(dims):
    torch.manual_seed(0)
    op = lambda x: x.flip(dims)
    x = torch.randn((2, 3))
    op_harness(op, x)


@pytest.mark.parametrize("with_clone", [True, False])
@pytest.mark.parametrize("with_detach", [True, False])
def test_detach_and_clone(with_clone, with_detach):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_layer = torch.nn.Linear(10, 10)
            self.second_layer = torch.nn.Linear(10, 10)

        def forward(self, x):
            out = self.first_layer(x)
            if with_clone:
                out = out.clone()
            if with_detach:
                out = out.detach()

            out = self.second_layer(out)
            return out

    model = Model()
    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.MSELoss(),
                                                   optimizer=torch.optim.SGD(
                                                       model.parameters(),
                                                       lr=0.01))

    target = torch.ones(10)
    input = torch.randn(10)

    bias_at_start = model.first_layer.bias.clone().data
    weight_at_start = model.first_layer.weight.clone().data

    for _ in range(100):
        _, _ = poptorch_model(input, target)

    if with_detach:
        assert (bias_at_start == model.first_layer.bias).all()
        assert (weight_at_start == model.first_layer.weight).all()
    else:
        assert (bias_at_start != model.first_layer.bias).all()
        assert (weight_at_start != model.first_layer.weight).all()


@helpers.printCapfdOnExit
def test_requires_grad_true(capfd):
    model = torch.nn.Linear(1, 1)
    poptorch_model = poptorch.inferenceModel(model)

    poptorch_model(torch.tensor([0.0], requires_grad=True))
    log = helpers.LogChecker(capfd)
    log.assert_contains("Input tensor has requires_grad=True set." +
                        "This tensor will be detached.")
