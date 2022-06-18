#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch
import helpers

non_differentiable_ops = [
    torch.ceil, torch.floor, torch.round, torch.sign, torch.trunc,
    torch.argmax, torch.argmin, torch.remainder, torch.floor_divide
]


def op_harness(trace_model,
               op,
               inputs,
               assert_func,
               test_training=False,
               out_fn=None):
    is_unary = len(inputs) == 1
    if not is_unary:
        assert len(inputs) == 2

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    if test_training and not op in non_differentiable_ops:
        model = helpers.ModelWithWeights(op, inputs[0].shape, out_fn)

        # Run on CPU.
        native_out, _ = model(tuple(inputs))

        # The LR should be large enough that a single training step will
        # definitely cause weights to change
        optim = torch.optim.AdamW(model.parameters(), lr=0.1)

        # Run on IPU.
        poptorch_model = poptorch.trainingModel(model,
                                                options=options,
                                                optimizer=optim)
        poptorch_out, _ = poptorch_model(tuple(inputs))

        # Training test - check weights have changed
        poptorch_model.assert_weights_changed()
    else:

        class Model(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op

        if is_unary:
            Model.forward = lambda self, x: self.op(x)
        else:
            Model.forward = lambda self, x, y: self.op(x, y)

        model = Model(op)

        # Run on CPU.
        native_out = model(*inputs)

        # Run on IPU.
        poptorch_model = poptorch.inferenceModel(model, options)
        poptorch_out = poptorch_model(*inputs)

    assert_func(native_out, poptorch_out)


unary_ops_float = [
    torch.abs,
    torch.acos,
    torch.acosh,
    torch.asin,
    torch.asinh,
    torch.atan,
    torch.atanh,
    # torch.angle,
    torch.ceil,
    torch.cos,
    torch.cosh,
    # torch.conj, torch.digamma
    torch.erf,
    torch.erfc,
    #torch.erfinv,
    torch.exp,
    torch.expm1,
    torch.floor,
    torch.frac,
    # torch.imag, torch.lgamma,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    # torch.logical_not, torch.mvlgamma,
    torch.neg,
    # torch.real,
    torch.reciprocal,
    torch.round,
    torch.rsqrt,
    torch.sigmoid,
    torch.sign,
    torch.sin,
    torch.sinh,
    torch.sqrt,
    torch.square,
    torch.tan,
    torch.tanh,
    torch.trunc,
]


@pytest.mark.parametrize("op", unary_ops_float)
@pytest.mark.parametrize("trace_model", [True, False])
def test_unary_ops_float(op, trace_model):
    torch.manual_seed(42)
    if not trace_model and op in [
            torch.acos, torch.acosh, torch.asin, torch.asinh, torch.atan,
            torch.atanh, torch.cos, torch.cosh, torch.erf, torch.erfc,
            torch.expm1, torch.frac, torch.log10, torch.log1p, torch.log2,
            torch.rsqrt, torch.reciprocal, torch.sin, torch.sinh, torch.tan
    ]:
        pytest.skip("TODO(T51159): No shape inference handler")

    input = torch.randn([1, 2, 10, 10])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(expected=native_out,
                                actual=poptorch_out,
                                atol=1e-03,
                                rtol=1e-03,
                                equal_nan=True)

    op_harness(trace_model, op, [input], assert_, test_training=True)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("exponent", [4.0, 3, 2.5])
@pytest.mark.parametrize("trace_model", [True, False])
def test_binary_pow(inplace, exponent, trace_model):
    torch.manual_seed(42)
    input = torch.randn([1, 2, 10, 200])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                equal_nan=True)

    def op(x):
        if inplace:
            # Although inplace would work, the native and poptorch output will
            # naturally not match as the input is changed
            x = x + 0
            return x.pow_(exponent)
        return torch.pow(x, exponent)

    op_harness(trace_model, op, [input], assert_)


unary_ops_int = [
    torch.bitwise_not,
]


@pytest.mark.parametrize("op", unary_ops_int)
@pytest.mark.parametrize("trace_model", [True, False])
def test_unary_ops_int(op, trace_model):
    torch.manual_seed(42)

    input = torch.randint(-1000, 1000, [1, 2, 10, 200])

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op, [input], assert_)


unary_ops_bool = [
    torch.bitwise_not,
]


@pytest.mark.parametrize("op", unary_ops_bool)
@pytest.mark.parametrize("trace_model", [True, False])
def test_unary_ops_bool(op, trace_model):
    torch.manual_seed(42)

    input = torch.randint(2, [1, 2, 10, 200]) > 0

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op, [input], assert_)


# Parameterize torch.clamp unittests for different supported overloads
clamp_inputs = [{"min": 0.2, "max": 0.8}, {"min": 0.2}, {"max": 0.8}]


@pytest.mark.parametrize("args", clamp_inputs)
@pytest.mark.parametrize("trace_model", [True, False])
def test_clamp(args, trace_model):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 10])

    def op_clamp(x):
        return x.clamp(**args)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op_clamp, [input], assert_, test_training=True)


@pytest.mark.parametrize("args", clamp_inputs)
@pytest.mark.parametrize("trace_model", [True, False])
def test_clamp_(args, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): RuntimeError: a leaf Variable that requires grad "
            "is being used in an in-place operation.")
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 10])

    def op_clamp_(x):
        return x.clamp_(**args)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op_clamp_, [input], assert_, test_training=True)


@pytest.mark.parametrize(
    "op",
    [torch.clamp_min, torch.clamp_min_, torch.clamp_max, torch.clamp_max_])
@pytest.mark.parametrize("trace_model", [True, False])
def test_clamp_min_max(op, trace_model):
    if not trace_model and op in [torch.clamp_max_, torch.clamp_min_]:
        pytest.skip(
            "TODO(T51159): RuntimeError: a leaf Variable that requires grad "
            "is being used in an in-place operation.")
    if not trace_model and op in [torch.clamp_max, torch.clamp_min]:
        pytest.skip("TODO(T51159): No shape inference handler for "
                    "aten::clamp_max.out / aten::clamp_min.out")

    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 10])

    def op_clamp(x):
        return op(x, 0.5)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op_clamp, [input], assert_, test_training=True)


clamp_int_inputs = [
    {
        "min": -4.5,
        "max": 5.5
    },
    {
        "min": -4.5
    },
    {
        "max": 5.5
    },
    {
        "min": -5,
        "max": 5
    },
    {
        "min": -5
    },
    {
        "max": 5
    },
]


@pytest.mark.parametrize("args", clamp_int_inputs)
def test_clamp_min_max_int(args):
    torch.manual_seed(42)

    t = torch.randint(-100, 100, (100, ))

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, **args)

    model = Model()
    ipu_model = poptorch.inferenceModel(model)

    helpers.assert_allequal(actual=ipu_model(t), expected=model(t))


binary_ops_float = [
    torch.add, torch.atan2, torch.div, torch.sub, torch.fmod,
    torch.floor_divide, torch.mul, torch.remainder, torch.true_divide
]


@pytest.mark.parametrize("op", binary_ops_float)
@pytest.mark.parametrize("trace_model", [True, False])
def test_binary_ops_float(op, trace_model):
    if not trace_model and op == torch.atan2:
        pytest.skip("TODO(T51159): No shape inference handler")
    if not trace_model and op == torch.amin:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::amin.out")

    torch.manual_seed(42)

    input1 = torch.randn([1, 2, 5, 1]) * 100.0
    input2 = torch.randn([1, 2, 5, 1]) * 10.0

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    op_harness(trace_model, op, [input1, input2], assert_, test_training=True)


binary_ops_basic_element_wise_float = [
    torch.add,
    torch.div,
    torch.sub,
    torch.mul,
]


@pytest.mark.parametrize("op", binary_ops_basic_element_wise_float)
@pytest.mark.parametrize("trace_model", [True, False])
def test_binary_ops_elementwise_edgecases(op, trace_model):
    torch.manual_seed(42)
    input1 = torch.randn([1, 2, 10, 10])
    input2 = torch.randn([1])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-04,
                                rtol=1e-04,
                                equal_nan=True)

    class Model(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

    # Constant on LHS
    Model.forward = lambda self, x, _y: self.op(x, 4.0)
    op_harness(trace_model,
               Model(op), [input1, input2],
               assert_,
               test_training=True)

    # Constant on RHS
    Model.forward = lambda self, x, _y: self.op(2.5, x)
    op_harness(trace_model,
               Model(op), [input1, input2],
               assert_,
               test_training=True)

    # Constant on LHS wrong type.
    Model.forward = lambda self, x, _y: self.op(x, 4)
    op_harness(trace_model,
               Model(op), [input1, input2],
               assert_,
               test_training=True)

    # Constant on RHS wrong type
    Model.forward = lambda self, x, _y: self.op(134, x)
    op_harness(trace_model,
               Model(op), [input1, input2],
               assert_,
               test_training=True)


binary_ops_basic_element_wise_bool = [
    torch.add,
    torch.mul,
]


@pytest.mark.parametrize("op", binary_ops_basic_element_wise_bool)
@pytest.mark.parametrize("trace_model", [True, False])
def test_binary_ops_elementwise_bools(op, trace_model):
    input1 = torch.tensor([False, True, False, True])
    input2 = torch.tensor([False, False, True, True])

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    class Model(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

    # Both bools
    Model.forward = lambda self, x, y: self.op(x, y)
    op_harness(trace_model, Model(op), [input1, input2], assert_)

    # Float on LHS
    Model.forward = lambda self, x, y: self.op(x.to(torch.float) + 1.0, y)
    op_harness(trace_model, Model(op), [input1, input2], assert_)

    # Float on RHS
    Model.forward = lambda self, x, y: self.op(x, y.to(torch.float) + 1.0)
    op_harness(trace_model, Model(op), [input1, input2], assert_)

    # Int on LHS
    Model.forward = lambda self, x, y: self.op(x.to(torch.int) + 1, y)
    op_harness(trace_model, Model(op), [input1, input2], assert_)

    # Int on RHS
    Model.forward = lambda self, x, y: self.op(x, y.to(torch.int) + 1)
    op_harness(trace_model, Model(op), [input1, input2], assert_)


@pytest.mark.parametrize("op", [torch.fmod, torch.remainder])
@pytest.mark.parametrize("trace_model", [True, False])
def test_modulo_mixed_sign(op, trace_model):
    input1 = torch.tensor([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0])
    input2 = torch.tensor([2.1, -3.4, 8.0, -2.1, 3.4, 5.0])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    op_harness(trace_model, op, [input1, input2], assert_)


def __and__(x, y):
    return x & y


def __or__(x, y):
    return x | y


def __xor__(x, y):
    return x ^ y


binary_op_int = [
    torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor, __and__, __or__,
    __xor__
]


@pytest.mark.parametrize("op", binary_op_int)
@pytest.mark.parametrize("trace_model", [True, False])
def test_binary_int_ops(op, trace_model):
    input1 = torch.tensor([-4, 7, 5, 4, -7, 8], dtype=torch.int)
    input2 = torch.tensor([2, -3, 8, -2, 3, 5], dtype=torch.int)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    op_harness(trace_model, op, [input1, input2], assert_)


# Poplar doesn't support binary ops on 8-bit integral types, but test we can
# pass the rest of them.
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("trace_model", [True, False])
def test_binary_int_op_types(dtype, trace_model):
    input1 = torch.tensor([-4, 7, 5, 4, -7, 8], dtype=dtype)
    input2 = torch.tensor([2, -3, 8, -2, 3, 5], dtype=dtype)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    op_harness(trace_model, torch.bitwise_and, [input1, input2], assert_)


binary_op_bool = [
    torch.bitwise_and,
    torch.bitwise_or,
    # torch.bitwise_xor, TODO(T43716)
    torch.logical_and,
    torch.logical_or,
    #torch.logical_xor TODO(T43716)
]


@pytest.mark.parametrize("op", binary_op_bool)
@pytest.mark.parametrize("trace_model", [True, False])
def test_binary_bool_ops(op, trace_model):
    input1 = torch.tensor([-4, 7, 5, 4, -7, 8]) > 0
    input2 = torch.tensor([2, -3, 8, -2, 3, 5]) > 0

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    op_harness(trace_model, op, [input1, input2], assert_)


# These functions support API 1 - op(input)
reduction_ops_api1 = [
    torch.amax,
    torch.amin,
    torch.argmax,
    torch.argmin,
    # torch.dist,
    torch.mean,
    torch.median,
    # torch.mode,
    torch.norm,
    torch.prod,
    #torch.std, torch.std_mean,
    torch.sum,
    #torch.unique, torch.unique_consecutive,torch.var, torch.var_mean,
]

# These functions support API 2 - op(input,dim,keep_dim)
reduction_ops_api2 = [
    torch.amax,
    torch.amin,
    torch.argmax,
    torch.argmin,
    # torch.dist,
    torch.mean,
    torch.median,
    # torch.mode,
    torch.norm,
    torch.prod,
    torch.logsumexp,  # logsumexp doesn't support API 1.
    #torch.std, torch.std_mean,
    torch.sum,
    #torch.unique, torch.unique_consecutive,torch.var, torch.var_mean,
]


@pytest.mark.parametrize("op", reduction_ops_api1)
@pytest.mark.parametrize("trace_model", [True, False])
def test_reduction_ops_float(op, trace_model):
    if not trace_model and op in [torch.amin]:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::amin.out")
    if not trace_model and op in [torch.amax]:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::amax.out")
    if not trace_model and op in [torch.median]:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::median")
    if not trace_model and op in [torch.norm]:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::resize_")
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 10])

    def assert_(native_out, poptorch_out):
        poptorch_out = poptorch_out.reshape(native_out.shape)
        if native_out.dtype == torch.float32:
            helpers.assert_allclose(actual=poptorch_out,
                                    expected=native_out,
                                    atol=1e-05,
                                    rtol=1e-05,
                                    equal_nan=True)
        else:
            helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op, [input], assert_, test_training=True)


@pytest.mark.parametrize("op", reduction_ops_api2)
@pytest.mark.parametrize("dim", range(4))
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("trace_model", [True, False])
def test_reduction_ops_float_api2(op, dim, keepdim, trace_model):
    if not trace_model and op in [torch.amin]:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::amin.out")
    if not trace_model and op in [torch.amax, torch.logsumexp]:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::amax.out")
    if not trace_model and op in [torch.median]:
        pytest.skip("TODO(T51159): No shape inference handler for "
                    "aten::median.dim_values")
    if not trace_model and op in [torch.norm]:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::resize_")
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 10])

    def operation(x):
        return op(x, dim=dim, keepdim=keepdim)

    def assert_(native_out, poptorch_out):
        if op is torch.median:
            # Median returns values and indices with API 2.
            helpers.assert_allclose(actual=poptorch_out[0],
                                    expected=native_out.values)
            helpers.assert_allequal(actual=poptorch_out[1].to(torch.int64),
                                    expected=native_out.indices)
        elif native_out.dtype == torch.float32:
            helpers.assert_allclose(actual=poptorch_out, expected=native_out)
        elif torch.numel(native_out) > 1:
            # Work around not returning longs from popart.
            helpers.assert_allequal(actual=poptorch_out.to(torch.int64),
                                    expected=native_out)
        else:
            helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    # This check must be repeated here because we need to check the op before we
    # wrap the function otherwise it won't match in the test harness
    test_training = not op in non_differentiable_ops
    out_fn = (lambda x: x.values) if op is torch.median else None
    op_harness(trace_model,
               operation, [input],
               assert_,
               test_training=test_training,
               out_fn=out_fn)


@pytest.mark.parametrize("op", [torch.min, torch.max])
@pytest.mark.parametrize("dim", range(3))
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("trace_model", [True, False])
def test_minmax_tuple_out(op, dim, keepdim, trace_model):
    if not trace_model and op == torch.max:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::max.dim")
    if not trace_model and op == torch.min:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::min.dim")
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 10])

    def operation(x):
        return op(x, dim=dim, keepdim=keepdim)

    def assert_(native_out, poptorch_out):
        assert isinstance(native_out, tuple) and isinstance(
            poptorch_out, tuple)
        assert len(native_out) == len(poptorch_out)
        for i, native in enumerate(native_out):
            helpers.assert_allclose(actual=poptorch_out[i], expected=native)

    out_fn = lambda x: x.values
    op_harness(trace_model,
               operation, [input],
               assert_,
               test_training=True,
               out_fn=out_fn)


# Interesting p-values for testing torch.norm(X, p=<>)
norm_pvals = ['fro', float('inf'), float('-inf'), 1, 1.0, 2, 2.0, 3, 3.0]


@pytest.mark.parametrize("p", norm_pvals)
@pytest.mark.parametrize("trace_model", [True, False])
def test_norm_p_values(p, trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): No shape inference handler for aten::resize_")
    torch.manual_seed(42)
    input = torch.randn([1, 2, 10, 10])

    def operation(x):
        return torch.norm(x, p=p)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, operation, [input], assert_, test_training=True)


comparison_ops = [
    # torch.allclose,     # Not supported in trace, seems to get optimized out.
    # torch.argsort,     # Not in Onnx. TODO(T23319)
    torch.eq,
    # torch.equal,       # Not supported as the return of trace in JIT.
    torch.ge,
    torch.gt,
    # torch.kthvalue,     # Not in Onnx.
    torch.le,
    torch.lt,
    torch.max,
    torch.min,
    torch.ne,
    # torch.sort,         # Not in Onnx (could be added via TopK if onnx supported TODO(T23319))
]


@pytest.mark.parametrize("op", comparison_ops)
@pytest.mark.parametrize("trace_model", [True, False])
def test_compare_operations(op, trace_model):
    torch.manual_seed(42)

    lhs = torch.randn([1, 2, 10, 200])
    rhs = torch.randn([1, 2, 10, 200])

    indices = torch.randint(0, 200, [30])

    # Make a few of the indices equal.
    for i in indices:
        lhs[0][0][0][i] = rhs[0][0][0][i]

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op, [lhs, rhs], assert_)

    if op not in (torch.min, torch.max):
        constant_rhs = lambda x: op(x, 0.34)
        op_harness(trace_model, constant_rhs, [lhs], assert_)


comparison_unity_nan_inf_ops = [
    # torch.isfinite, torch.isinf,  # Not in Onnx
    torch.isnan,
]


@pytest.mark.parametrize("op", comparison_unity_nan_inf_ops)
@pytest.mark.parametrize("trace_model", [True, False])
def test_compare_unity_nan_inf_ops(op, trace_model):
    torch.manual_seed(42)

    input = torch.tensor([
        1.0,
        float('inf'), 2.0,
        float('-inf'),
        float('nan'),
        float('-nan'), 13.0
    ])

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, op, [input], assert_)


comparison_unity = [torch.max, torch.min]


@pytest.mark.parametrize("op", comparison_unity)
@pytest.mark.parametrize("trace_model", [True, False])
def test_compare_unity_operations(op, trace_model):
    if not trace_model and op == torch.max:
        pytest.skip("TODO(T51159): No shape inference handler for aten::max")
    if not trace_model and op == torch.min:
        pytest.skip("TODO(T51159): No shape inference handler for aten::min")

    torch.manual_seed(42)
    input = torch.randn([1, 2, 10, 10])

    def operation(x):
        return op(x)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, operation, [input], assert_, test_training=True)


# Support other arguments. TODO(T23319)
@pytest.mark.parametrize("trace_model", [True, False])
def test_topk(trace_model):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 10])

    def operation(x):
        return torch.topk(x, k=10, dim=-1)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out[0],
                                expected=native_out.values)
        helpers.assert_allequal(actual=poptorch_out[1],
                                expected=native_out.indices)

    out_fn = lambda x: x.values
    op_harness(trace_model,
               operation, [input],
               assert_,
               test_training=True,
               out_fn=out_fn)


types = [torch.float32, torch.int32]


@pytest.mark.parametrize("ty", types)
@pytest.mark.parametrize("trace_model", [True, False])
def test_constant_arrays(ty, trace_model):
    torch.manual_seed(42)

    input = torch.randn([10]).to(ty)

    def operation(x):
        constant_tensor = torch.tensor([1, -2, -3, 4, 5, 6, 7, -8, 9, -10],
                                       dtype=ty)
        return torch.sub(x, constant_tensor)

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, operation, [input], assert_)


@pytest.mark.parametrize("ty", types)
@pytest.mark.parametrize("trace_model", [True, False])
def test_big_constant_arrays_sliced(ty, trace_model):
    torch.manual_seed(42)

    input = torch.randn([1]).to(ty)

    def operation(x):
        big_array = torch.tensor(
            [[
                155, 229, 322, 453, 655, 888, 1128, 1694, 2036, 2502, 3089,
                3858, 4636, 5883, 7375, 9172, 10149, 12462, 15113, 17660,
                21157, 24747, 27980, 31506, 35713, 41035, 47021, 43, 59138,
                63927, 69176, 74386, 80589, 86498, 92472, 97689, 45, -424, 5,
                6, 435, 124632, 128948, 132547, 135586, 42, 5, 147577, 5
            ],
             [
                 2, 1, 1, 3, 45, 46, 46, 83, 149, 160, 276, 414, 523, 589, 622,
                 724, 724, 1045, 1045, 1439, 24, 2335, 2749, 2941, 4025, 4440,
                 4440, 24, 7024, 7024, 8326, 9362, 10361, 10950, 12384, 13030,
                 -8, 324, 425, 67, -245, -2425, 21815, 22837, 24392, 324, 234,
                 2435, 4325
             ],
             [
                 3, 7, 10, 12, 17, 21, 29, 34, 52, 79, 107, 148, 197, 233, 366,
                 463, 631, 827, -2344, -2, 1441, 1809, 2158, 2503, 2978, 3405,
                 4032, -324, 5664, 45, 53, -25, 8215, 9134, 10023, 10779,
                 -2345, 4, 13155, 5, 98754, 143535, 245232, 16523, 17127, 2,
                 42, 5, 19468
             ]],
            dtype=ty)
        return x * big_array[0]

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, operation, [input], assert_)


# Parametrize input tensor shapes for addcdiv to make sure broadcasting works.
broadcastable_shapes = [
    ((3, 1), (3, 1), (3, 1)),
    ((1, 3), (3, 1), (1, 3)),
    ((5, 3), (5, 1), (1, 3)),
    ((1, ), (3, 1), (2, )),
]


@pytest.mark.parametrize("shapes", broadcastable_shapes)
@pytest.mark.parametrize("scale", [0.35, 4.91, 12.0, -0.53, -3.45, -9.0, 0.0])
@pytest.mark.parametrize("trace_model", [True, False])
def test_addcdiv(shapes, scale, trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def forward(self, tensor0, tensor1, tensor2):
            return torch.addcdiv(
                tensor0,
                tensor1,
                tensor2,
                value=scale,
            )

    t0 = torch.randn(shapes[0])
    t1 = torch.randn(shapes[1])
    t2 = torch.randn(shapes[2])

    model = Model()
    native_out = model(t0, t1, t2)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(t0, t1, t2)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


cross_shapes = [(3, 4, 5, 6), (4, 3, 5, 6), (4, 5, 3, 6), (4, 5, 6, 3),
                (6, 3, 3, 5)]


@pytest.mark.parametrize("shape", cross_shapes)
@pytest.mark.parametrize("trace_model", [True, False])
def test_cross_shape(shape, trace_model):
    if not trace_model:
        pytest.skip("TODO(T51159): No shape inference handler for aten::cross")
    torch.manual_seed(42)

    x = torch.randn(shape)
    y = torch.randn(shape)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, torch.cross, [x, y], assert_, test_training=True)


@pytest.mark.parametrize("axis", range(0, 4))
@pytest.mark.parametrize("trace_model", [True, False])
def test_cross_axis(axis, trace_model):
    if not trace_model:
        pytest.skip("TODO(T51159): No shape inference handler for aten::cross")
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, x, y):
            return torch.cross(x, y, self.axis)

    x = torch.randn(3, 3, 3, 3)
    y = torch.randn(3, 3, 3, 3)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, Model(axis), [x, y], assert_, test_training=True)


@pytest.mark.parametrize(
    "trace_model, params",
    [
        # trace_model, dims?, unbiased
        (True, (True, )),
        (False, (False, )),
        (True, ([2], False)),
        (False, ([0, 1], True)),
        (True, ([0, 2], False)),
    ])
@pytest.mark.parametrize(
    "op", [torch.var, torch.var_mean, torch.std, torch.std_mean])
def test_var_std(op, trace_model, params):
    torch.manual_seed(42)

    x = torch.randn(3, 4, 5)
    model = lambda x: op(x, *params)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, model, [x], assert_)


@pytest.mark.parametrize("trace_model", [True, False])
@pytest.mark.parametrize("axis", range(0, 4))
@pytest.mark.parametrize("descending", [True, False])
def test_argsort(trace_model, axis, descending):
    torch.manual_seed(42)
    input = torch.randn([3, 4, 5, 5])

    def operation(x):
        return torch.argsort(x, dim=axis, descending=descending)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    op_harness(trace_model, operation, [input], assert_)
