#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import copy
from functools import partial
import re
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


def zeros_and_ones_harness(model, dtype, is_like, trace_model):
    assert dtype in [torch.float16, torch.float32, torch.int32, torch.bool]
    torch.manual_seed(42)

    # Calculating with ints/bools does not produce meaningful gradients
    test_training = not dtype in (torch.int32, torch.bool)

    inputs = [torch.tensor([1], dtype=dtype)]
    if is_like:
        inputs.append(torch.empty(3, 5, 1))
    inputs = tuple(inputs)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    if test_training:
        out_fn = lambda out: out[0]
        model = helpers.ModelWithWeights(model, inputs[0].shape, out_fn=out_fn)
        # We need to copy the model to use the original weights for native comparison
        model_copy = copy.deepcopy(model)
        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model, options)
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
        poptorch_model = poptorch.inferenceModel(model, options)
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


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("dtype", zeros_and_ones_dtypes)
def test_zeros_and_ones(dtype, trace_model):
    class Model(torch.nn.Module):
        def forward(self, z):
            x = torch.zeros(3, 5, 1, dtype=dtype)
            y = torch.ones(3, 5, 1, dtype=dtype)

            return (x * y) + z, (y + x) + z

    zeros_and_ones_harness(Model(), dtype, False, trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("dtype", zeros_and_ones_dtypes)
def test_new_zeros_and_new_ones(dtype, trace_model):
    class Model(torch.nn.Module):
        def forward(self, z):
            x = z.new_zeros(3, 5, 1)
            y = z.new_ones(3, 5, 1)

            return (x * y) + z, (y + x) + z

    zeros_and_ones_harness(Model(), dtype, False, trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("dtype", zeros_and_ones_dtypes)
def test_zeros_like_and_ones_like(dtype, trace_model):
    class Model(torch.nn.Module):
        def forward(self, z, t):
            x = torch.zeros_like(t, dtype=dtype)
            y = torch.ones_like(t, dtype=dtype)

            return (x * y) + z, (y + x) + z

    zeros_and_ones_harness(Model(), dtype, True, trace_model)


def fuzzy_compare_exceptions(e_cpu, e_ipu):
    """Compares error messages from CPU and IPU implementations
    if they do not match a fuzzy comparison (all words in the CPU exception
    are also in the IPU exception) an error is raised.
    """
    e_ipu_words = {word: i for i, word in enumerate(str(e_ipu).split())}
    if not all([
            e_ipu_words.get(word, -1) >= i
            for i, word in enumerate(str(e_cpu).split())
    ]):
        raise ValueError("CPU and IPU error messages did not match: " +
                         f"'{e_cpu}' not in '{e_ipu}'") from e_ipu
    print(f"CPU and IPU error messages did match: '{e_cpu}' in '{e_ipu}'")


def op_harness(op,
               *inputs,
               test_training=True,
               assert_fn=None,
               out_fn=None,
               native_out=None,
               fuzzy_errors=False,
               trace_model=True,
               allow_native_errors=True):
    """The op harness allows to test the native torch API against poptorch.

    This function wraps an operation into a model and allows training and
    inference comparisons between py and poptorch.
    This function returns without errors when tensors are almost equal
    or the IPU and CPU implementation provide the same error messages.
    """

    def exception_catcher(model, *inputs, can_raise_exception=True):
        __tracebackhide__ = True  # pylint: disable=W0612
        op_raises_exception = False
        try:
            if test_training:
                native_out, _ = model(*inputs)
            else:
                native_out = model(*inputs)
        except Exception as e:  # pylint: disable=W0703
            if not can_raise_exception:
                raise
            native_out = ("error", e)
            op_raises_exception = True
            assert not poptorch.poptorch_core.isCompilingWithDispatcher(), (
                "[Internal] Clean up failed: dispatcher still active")
        return native_out, op_raises_exception

    if assert_fn is None:

        def assert_fn(native_out, poptorch_out):
            if isinstance(native_out, tuple):
                for native, pop in zip(native_out, poptorch_out):
                    helpers.assert_allclose(expected=native, actual=pop)
            else:
                helpers.assert_allclose(expected=native_out,
                                        actual=poptorch_out)

    op_raises_exception = False
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    if test_training:
        # Set a fixed seed for the weights of the model
        torch.manual_seed(42)
        model = helpers.ModelWithWeights(op, inputs[0].shape, out_fn=out_fn)

        # Run on CPU.
        if native_out is None:
            native_out, op_raises_exception = exception_catcher(model, inputs)

            # native_out could be an alias of the input and so modified by
            # the poptorch_model, except if its an error
            if op_raises_exception:
                if not allow_native_errors:
                    raise native_out[1]
            elif isinstance(native_out, tuple):
                # pylint: disable=E1101
                native_out = tuple([n.clone().detach() for n in native_out])
            else:
                native_out = native_out.clone().detach()
        else:
            op_raises_exception = isinstance(
                native_out, tuple) and native_out[0] == "error"

        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model, options=options)
        poptorch_out, ipu_raises = exception_catcher(
            poptorch_model, inputs, can_raise_exception=op_raises_exception)

        # Training test - check weights changed if no error was thrown
        try:
            poptorch_model.assert_weights_changed()
            assert not op_raises_exception, (
                "Weights changed despite errors being "
                "thrown in IPU evaluation.")
        except AssertionError:
            if not op_raises_exception:
                raise
    else:
        model = torch.nn.Module()
        model.forward = op

        # Run on CPU.
        if native_out is None:
            native_out, op_raises_exception = exception_catcher(model, *inputs)
            if op_raises_exception and not allow_native_errors:
                raise native_out[1]
        else:
            op_raises_exception = isinstance(
                native_out, tuple) and native_out[0] == "error"

        poptorch_model = poptorch.inferenceModel(model, options)
        # Run on IPU.
        poptorch_out, ipu_raises = exception_catcher(
            poptorch_model, *inputs, can_raise_exception=op_raises_exception)

    # Compare outputs
    if not ipu_raises and op_raises_exception:
        _, cpu_error = native_out
        raise RuntimeError(
            "The torch and poptorch API do not match, " +
            "poptorch returned without error while torch failed" +
            f" with {cpu_error}") from cpu_error
    if fuzzy_errors and op_raises_exception:
        fuzzy_compare_exceptions(native_out[1], poptorch_out[1])
    elif op_raises_exception:
        _, cpu_error = native_out
        _, ipu_error = poptorch_out
        with pytest.raises(type(cpu_error),
                           match="^" + re.escape(f"{cpu_error}") + "$"):
            raise ipu_error
    else:
        assert_fn(native_out, poptorch_out)


class TestOpHarness:
    """Test the exception matching functionality of the op_harness function."""
    exact_error_check = "Regex pattern.*does not match"
    fuzzy_error_check = "CPU and IPU error messages did not match"
    op_harness = op_harness

    @pytest.fixture(autouse=True, params=[True, False])
    def training(self, request, monkeypatch):
        monkeypatch.setattr(self, "op_harness",
                            partial(op_harness, test_training=request.param))

    @pytest.fixture(autouse=True, params=[True, False])
    def trace_model(self, request, monkeypatch):
        monkeypatch.setattr(self, "op_harness",
                            partial(op_harness, trace_model=request.param))

    def test_fuzzy_error_mismatch(self):
        x = torch.randn(2, 3)

        def op(x):
            raise ValueError("Hi")

        with pytest.raises(ValueError, match=self.fuzzy_error_check):
            self.op_harness(op,
                            x,
                            native_out=("error", ValueError("Hey")),
                            fuzzy_errors=True)

    def test_error_mismatch(self):
        x = torch.randn(2, 3)

        def op(x):
            raise ValueError("Hi")

        with pytest.raises(AssertionError, match=self.exact_error_check):
            self.op_harness(op, x, native_out=("error", ValueError("Hey")))

    def test_exact_match(self):
        x = torch.randn(2, 3)

        def op(x):
            raise ValueError("Hi")

        self.op_harness(op, x)

    def test_fuzzy_match(self):
        x = torch.randn(2, 3)

        def op(x):
            raise ValueError("Hi Hey")

        self.op_harness(op,
                        x,
                        native_out=("error", ValueError("Hey")),
                        fuzzy_errors=True)

    def test_fuzzy_mismatch(self):
        x = torch.randn(2, 3)

        def op(x):
            raise ValueError("Hi")

        with pytest.raises(ValueError, match=self.fuzzy_error_check):
            self.op_harness(op,
                            x,
                            native_out=("error", ValueError("Hey Hi")),
                            fuzzy_errors=True)

    def test_reject_fuzzy_match_without_fuzzy_option(self):
        x = torch.randn(2, 3)

        def op(x):
            raise ValueError("Hi Hey")

        with pytest.raises(AssertionError, match=self.exact_error_check):
            self.op_harness(op, x, native_out=("error", ValueError("Hey")))

    def test_reject_exception_if_not_native(self):
        x = torch.randn(2, 3)
        error = ValueError("Hi Hey")

        def op(x):
            raise error

        with pytest.raises(type(error), match=f"{error}"):
            self.op_harness(op, x, native_out=(1))

    def test_no_ipu_exception_with_native_exception(self):
        x = torch.randn(2, 3)
        error = ValueError("Hi Hey")

        def op(x):
            return torch.roll(x, 1)

        with pytest.raises(RuntimeError, match=f"{error}"):
            self.op_harness(op, x, native_out=("error", error))


# Note: Many of the following operations don't depend on the values of the tensors
# but we still need to fix the random seed for any op with randomly generated values
# so that it's guaranteed that weights change after one training step


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize(
    "dtypes", [
        [torch.float] * 3,
        [torch.int] * 3,
        [torch.int, torch.float],
        [torch.float, torch.int],
    ],
    ids=["all_floats", "all_ints", "int,float", "float,int"])
def test_cat(dim, trace_model, dtypes):
    torch.manual_seed(42)
    # Cannot control the type of the first tensor as it needs to be
    # torch.float32 to be a valid input to the Linear layer used in
    # op_harness.
    first_input = torch.randn(2, 3)
    tensors = [torch.randn(2, 3).to(dtype=dtype) for dtype in dtypes]

    op = lambda *xs: torch.cat(xs, dim=dim)
    op_harness(op,
               first_input,
               *tensors,
               allow_native_errors=False,
               trace_model=trace_model)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("trace_model", [True, False])
def test_cat_transpose(dim, trace_model):
    """This combination of ops without ImplicitCasting causes the code
    to crash out."""
    if not trace_model:
        pytest.skip("TODO(T51159): Only Tensors of floating point and complex "
                    "dtype can require gradients")
    torch.manual_seed(42)
    floatTensor = torch.randn(2, 3).to(dtype=torch.float)
    intTensor = torch.randn(2, 3).to(dtype=torch.int)

    op = lambda floatTensor, intTensor: torch.cat((intTensor, floatTensor),
                                                  dim=dim).transpose(1, 0)

    op_harness(op,
               floatTensor,
               intTensor,
               allow_native_errors=False,
               trace_model=trace_model)


@pytest.mark.parametrize("dim_size", [11, 12, 13])
@pytest.mark.parametrize("trace_model", [True, False])
def test_chunk(dim_size, trace_model):
    torch.manual_seed(42)
    x = torch.randn(dim_size)

    op = lambda x: torch.chunk(x, 6)

    op_harness(op, x, out_fn=lambda x: x[0], trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_cat_chunk_slice(trace_model):
    def forward(x, mems):
        index = 8
        cat = torch.cat([mems, x], 0)
        split, _ = torch.chunk(cat, 2, dim=2)
        split2 = split[index:]
        return split2

    mems = torch.randn(1600, 1, 10, 10, 5)
    x = torch.randn(8, 1, 10, 10, 5)

    op = forward
    op_harness(op, x, mems, test_training=False, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_cat_chunk_slice_multiple_slices(trace_model):
    def forward(x, mems):
        index = 8
        cat = torch.cat([mems, x], 0)
        _, _, split2, _, _ = torch.chunk(cat, 5, dim=2)
        split5 = split2[index:]
        return split5

    mems = torch.randn(1600, 1, 10, 10, 5)
    x = torch.randn(8, 1, 10, 10, 5)

    op = forward
    op_harness(op, x, mems, test_training=False, trace_model=trace_model)


def fast_gather_last_dim(data, idx):
    assert poptorch.ipuHardwareIsAvailable(), \
           "Hardware IPU needed to compile this FastGatherLastDim custom op"
    out = None
    if poptorch.isRunningOnIpu():
        target = torch.zeros(idx.shape).type_as(data)
        target.requires_grad_()
        o = poptorch.custom_op([data, idx],
                               "FastGatherLastDim",
                               "poptorch.custom_ops",
                               1,
                               example_outputs=[target],
                               attributes={})
        out = o[0]
    else:
        out = torch.gather(data, -1, idx)
    return out


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_fastgather_3dim(trace_model):
    torch.manual_seed(42)
    shape = (9, 11, 6)
    input = torch.randn(shape)
    indices = torch.randint(0, 6, shape)
    op_harness(fast_gather_last_dim, input, indices, trace_model=trace_model)

    # Gather index last dim smaller than input last dim
    indices = torch.randint(0, 6, (9, 11, 3))
    op_harness(fast_gather_last_dim, input, indices, trace_model=trace_model)

    # Gather index different shape should fail
    indices = torch.randint(0, 6, (9, 1, 6))
    with pytest.raises(poptorch.poptorch_core.Error):
        op_harness(fast_gather_last_dim,
                   input,
                   indices,
                   trace_model=trace_model)

    # Gather index different rank should fail
    indices = torch.randint(0, 6, (11, 6))
    with pytest.raises(poptorch.poptorch_core.Error):
        op_harness(fast_gather_last_dim,
                   input,
                   indices,
                   trace_model=trace_model)


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2])
@pytest.mark.parametrize("larger_index", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_gather_3dim(dim, larger_index, trace_model):
    torch.manual_seed(42)
    shape = (9, 11, 6)
    input = torch.randn(shape)

    indices = torch.randint(0, 6, shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices, trace_model=trace_model)

    small_shape = (7, 9, 5)
    if larger_index:
        larger_dims = list(small_shape)
        larger_dims[dim] = shape[dim] + 1
        small_shape = tuple(larger_dims)
    indices = torch.randint(0, 6, small_shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices, trace_model=trace_model)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("larger_index", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_gather_4dim(dim, larger_index, trace_model):
    torch.manual_seed(42)
    shape = (5, 8, 6, 7)
    input = torch.randn(shape)

    indices = torch.randint(0, 5, shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices, trace_model=trace_model)

    small_shape = (4, 5, 2, 6)
    if larger_index:
        larger_dims = list(small_shape)
        larger_dims[dim] = shape[dim] + 1
        small_shape = tuple(larger_dims)
    indices = torch.randint(0, 5, small_shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices, trace_model=trace_model)


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("larger_index", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_gather_5dim(dim, larger_index, trace_model):
    torch.manual_seed(42)
    shape = (3, 3, 3, 3, 3)
    input = torch.randn(shape)

    indices = torch.randint(0, 3, shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices, trace_model=trace_model)

    small_shape = (2, 2, 2, 2, 2)
    if larger_index:
        larger_dims = list(small_shape)
        larger_dims[dim] = shape[dim] + 1
        small_shape = tuple(larger_dims)
    indices = torch.randint(0, 3, small_shape)
    op = lambda x, y: torch.gather(x, dim, y)
    op_harness(op, input, indices, trace_model=trace_model)


@pytest.mark.parametrize("dim", range(-3, 3))
@pytest.mark.parametrize("trace_model", [True, False])
def test_scatter(dim, trace_model):
    torch.manual_seed(42)
    dim_length = 3
    shape = (dim_length, ) * 3

    input = torch.randn(shape)
    indices = torch.randint(dim_length, shape)
    source = torch.randn(shape)

    op = lambda inp, idx, src: inp.scatter(dim, idx, src)
    op_harness(op, input, indices, source, trace_model=trace_model)


@pytest.mark.parametrize("dim", range(-3, 3))
@pytest.mark.parametrize("trace_model", [True, False])
def test_scatter_(dim, trace_model):
    torch.manual_seed(42)
    dim_length = 3
    shape = (dim_length, ) * 3

    input = torch.randn(shape)
    indices = torch.randint(dim_length, shape)
    source = torch.randn(shape)

    op = lambda inp, idx, src: inp.scatter_(dim, idx, src)
    op_harness(op, input, indices, source, trace_model=trace_model)


@pytest.mark.parametrize("dim", range(-3, 3))
@pytest.mark.parametrize("trace_model", [True, False])
def test_scatter_scalar(dim, trace_model):
    torch.manual_seed(42)
    dim_length = 3
    shape = (dim_length, ) * 3

    input = torch.randn(shape)
    indices = torch.randint(dim_length, shape)
    source = 5.0

    op = lambda inp, idx: inp.scatter(dim, idx, source)
    op_harness(op, input, indices, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_reshape(trace_model):
    op = lambda x: torch.reshape(x, (1, 1, 2, 2))

    x = torch.arange(4.)

    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("split_size_or_sections",
                         (1, 5, 6, 20, [10, 10], [19, 1]))
@pytest.mark.parametrize("trace_model", [True, False])
def test_split(split_size_or_sections, trace_model):
    torch.manual_seed(42)
    x = torch.randn(20, 10)
    op = lambda x: torch.split(x, split_size_or_sections)

    op_harness(op, x, out_fn=lambda x: x[0], trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_split_singleton(trace_model):
    torch.manual_seed(42)
    x = torch.randn(1, 4, 3, 1)
    op = lambda x: torch.split(x, 1, 1)[0]

    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_squeeze(inplace, trace_model):
    torch.manual_seed(42)
    x = torch.randn(1, 1, 5, 1, 10, 1)

    def f(t):
        if inplace:
            t.squeeze_()
            return t
        return torch.squeeze(t)

    op_harness(f, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_t(trace_model):
    torch.manual_seed(42)
    x = torch.randn(20, 10)

    op_harness(torch.t, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_transpose(trace_model):
    torch.manual_seed(42)
    x = torch.randn(3, 2, 5, 2)
    op = lambda x: torch.transpose(x, 3, 0)

    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_transpose_negative_dims(trace_model):
    torch.manual_seed(42)
    x = torch.randn(3, 2, 5, 2)
    y = torch.randn(2, 2, 5, 3)
    op = lambda x, y: torch.transpose(x, -1, 0) + y

    op_harness(op, x, y, trace_model=trace_model, test_training=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_numpy_T(trace_model):
    torch.manual_seed(42)
    op = lambda x: x.T

    x = torch.randn(3, 2, 5, 4)
    op_harness(op, x, trace_model=trace_model)

    x = torch.randn(5)
    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_unsqueeze(trace_model):
    torch.manual_seed(42)
    x = torch.randn(3, 2, 5, 2)
    op = lambda x: torch.unsqueeze(x, 1)

    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_expand(trace_model):
    torch.manual_seed(42)
    x = torch.randn(3, 1)
    op = lambda x: x.expand(3, 4)

    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_expand_preserve_dim(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): Invalid tensor shape: dimension 1 is negative (-1)")
    torch.manual_seed(42)
    x = torch.randn(1, 1, 100)
    op = lambda x: x.expand(2, -1, -1)

    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_expand_as(trace_model):
    torch.manual_seed(42)
    x = torch.randn(3, 1)
    y = torch.randn(3, 4)
    op = lambda x, y: x.expand_as(y)

    op_harness(op, x, y, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_flatten(trace_model):
    torch.manual_seed(42)
    x = torch.randn(3, 1)

    op_harness(torch.flatten, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_view(trace_model):
    torch.manual_seed(42)
    x = torch.randn(30, 5)
    op = lambda x: x.view((15, 2, 5))

    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 2), (2, 3, 4)])
def test_size(input_shapes, trace_model):
    x = torch.ones(*input_shapes)
    # Use size as input to another operation to workaround pruning error
    op = lambda x: x.view(x.size())

    op_harness(op, x, trace_model=trace_model)


input_shapes = [(1, 4, 5), (2, ), (2, 2), (2, 3, 4, 1, 3, 4)]
dtypes = [torch.float, torch.float16, torch.int32]


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("input_shapes", input_shapes)
@pytest.mark.parametrize("t", dtypes)
@pytest.mark.parametrize("trace_model", [True, False])
def test_fill(capfd, input_shapes, t, trace_model):
    float_test_num = 1.9375

    def op(x):
        value = 42 if x.dtype == torch.int32 else float_test_num
        x = x + 0  # Ensure x is not modified in place
        # Add zero to all results to avoid pruning the whole graph
        return x.fill_(value) + 0, torch.full_like(x, value) + 0, torch.full(
            input_shapes, value, dtype=x.dtype) + 0, torch.ones_like(x) + 0

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
               native_out=native_out,
               trace_model=trace_model)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_no_matches("expand")


@pytest.mark.parametrize("input_shapes", input_shapes)
@pytest.mark.parametrize("value", [0.666, -4.32, float("Inf"), float("-Inf")])
@pytest.mark.parametrize("trace_model", [True, False])
def test_masked_fill(input_shapes, value, trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, x):
            fill_result = x.masked_fill(x > 0.5, value)
            where_result = torch.where(x > 0.5, x, torch.tensor(value))
            return fill_result, where_result

    x = torch.randn(*input_shapes)
    op_harness(Model(), x, out_fn=lambda x: x[0], trace_model=trace_model)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (3, 4), (1, 3, 4)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("trace_model", [True, False])
def test_stack(input_shapes, dim, trace_model):
    torch.manual_seed(42)

    if dim > len(input_shapes):
        pytest.skip()

    op = lambda *xs: torch.stack(xs, dim=dim)
    inputs = [torch.randn(*input_shapes) for _ in range(3)]

    op_harness(op, *inputs, trace_model=trace_model)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dims",
                         [[1], [3], [2, 1], [2, 3], [1, 1, 1], [3, 2, 4]])
@pytest.mark.parametrize("trace_model", [True, False])
def test_repeat(input_shapes, dims, trace_model):
    if len(dims) < len(input_shapes):
        pytest.skip(
            "Number of dimensions of repeat dims can not be smaller than number"
            " of dimensions of tensor.")

    torch.manual_seed(42)

    op = lambda x: x.repeat(dims)
    a = torch.randn(*input_shapes)

    op_harness(op, a, trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_repeat_training_input(trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

    native_out, _ = model(input)
    poptorch_out, _ = poptorch_model(input)

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.int])
@pytest.mark.parametrize("trace_model", [True, False])
def test_clone_one(input_shapes, dtype, trace_model):
    torch.manual_seed(42)

    op = lambda x: x.clone()

    x = torch.randn(*input_shapes)

    def assert_fn(native_out, poptorch_out):
        for pop, native in zip(poptorch_out, native_out):
            assert native.dtype == pop.dtype
            helpers.assert_allclose(expected=native, actual=pop)

    # Calculating with integers does not produce meaningful gradients
    test_training = dtype is torch.float
    op_harness(op,
               x,
               test_training=test_training,
               assert_fn=assert_fn,
               trace_model=trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_clone_two(trace_model):
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

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(dummy_x.clone(), dummy_y.clone(),
                                  dummy_z.clone())

    helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("input_shapes", [(1, ), (2, ), (2, 3), (1, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.half, torch.int])
@pytest.mark.parametrize("trace_model", [True, False])
def test_copy_(input_shapes, dtype, trace_model):
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
    op_harness(op,
               x,
               y,
               test_training=test_training,
               assert_fn=assert_fn,
               trace_model=trace_model)


@pytest.mark.parametrize("shifts,dims", [(1, 0), (-1, 0), (10, 1), (-10, 1),
                                         (0, 2), ((1, 1), (0, 1)),
                                         ((1, -1), (1, 2)), ((-3, -4), (0, 2)),
                                         ((1, 2, 3), (0, 1, 2)),
                                         ((-1, -2, -3), (0, 1, 2)), (5, None),
                                         (-3, None), (1, -1), (1, -3), (1, -4),
                                         (1, 3)])
@pytest.mark.parametrize("trace_model", [True, False])
def test_roll(shifts, dims, trace_model):
    torch.manual_seed(0)
    op = lambda x: x.roll(shifts, dims)
    x = torch.randn((2, 3, 4))
    op_harness(op, x, fuzzy_errors=True, trace_model=trace_model)


@pytest.mark.parametrize("dims", [0, 1, -1])
@pytest.mark.parametrize("trace_model", [True, False])
def test_flip(dims, trace_model):
    torch.manual_seed(0)
    op = lambda x: x.flip(dims)
    x = torch.randn((2, 3))
    op_harness(op, x, trace_model=trace_model)


@pytest.mark.parametrize("with_clone", [True, False])
@pytest.mark.parametrize("with_detach", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_detach_and_clone(with_clone, with_detach, trace_model):
    if not trace_model and with_detach:
        pytest.skip("TODO(T51159): assert tensor(False)")
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_layer = torch.nn.Linear(10, 10)
            self.second_layer = torch.nn.Linear(10, 10)
            self.loss = torch.nn.MSELoss()

        def forward(self, x, target):
            out = self.first_layer(x)
            if with_clone:
                out = out.clone()
            if with_detach:
                out = out.detach()

            out = self.second_layer(out)
            loss = self.loss(out, target)
            return out, loss

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model,
                                            options=options,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(), lr=0.01))

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
@pytest.mark.parametrize("trace_model", [True, False])
def test_requires_grad_true(capfd, trace_model):
    model = torch.nn.Linear(1, 1)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    poptorch_model(torch.tensor([0.0], requires_grad=True))
    log = helpers.LogChecker(capfd)
    log.assert_contains(
        "Input tensor has requires_grad=True set. " +
        "This tensor will be detached because backward pass via " +
        "inputs is not supported.")


@pytest.mark.parametrize("trace_model", [True, False])
def test_arange(trace_model):
    class ArangeModel(torch.nn.Module):
        def forward(self, a, b):
            def f(x, y):
                return torch.arange(20) + x, y

            return poptorch.for_loop(5, f, [a, b])

    model = ArangeModel()
    a = torch.randn(20)
    b = torch.randn(20)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(a, b)

    helpers.assert_allclose(actual=poptorch_out,
                            expected=[torch.arange(20) * 5 + a, b])


@pytest.mark.parametrize("input_shape,dim,size,step",
                         [((7, ), 0, 2, 1), ((7, ), 0, 2, 2),
                          ((10, ), 0, 2, 2), ((10, ), 0, 2, 1),
                          ((5, 5), 0, 2, 2), ((5, 5), 1, 2, 2),
                          ((3, 2, 1), 0, 2, 2), ((10, 10, 10), 1, 5, 2)])
@pytest.mark.parametrize("trace_model", [True, False])
def test_unfold(input_shape, dim, size, step, trace_model):
    torch.manual_seed(0)
    op = lambda x: x.unfold(dim, size, step)
    x = torch.randn(input_shape)
    op_harness(op, x, trace_model=trace_model)
