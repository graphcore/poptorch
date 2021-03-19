#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch
import helpers


def unary_op_harness(op, input, assert_func):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x):
            return self.op(x)

    model = Model(op)

    # Run on CPU.
    native_out = model(input)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    assert_func(native_out, poptorch_out)


def binary_op_harness(op, input1, input2, assert_func):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x, y)

    model = Model(op)

    # Run on CPU.
    native_out = model(input1, input2)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input1, input2)

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
def test_unary_ops_float(op):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(expected=native_out,
                                actual=poptorch_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    unary_op_harness(op, input, assert_)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("exponent", [4.0, 3, 2.5])
def test_binary_pow(inplace, exponent):
    torch.manual_seed(42)
    input1 = torch.randn([1, 2, 10, 200])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    def op(x):
        if inplace:
            x = x + 0  # Ensure input is not modified in place
            return x.pow_(exponent)
        return torch.pow(x, exponent)

    unary_op_harness(op, input1, assert_)


unary_ops_int = [
    torch.bitwise_not,
]


@pytest.mark.parametrize("op", unary_ops_int)
def test_unary_ops_int(op):
    torch.manual_seed(42)

    input = torch.randint(-1000, 1000, [1, 2, 10, 200])

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    unary_op_harness(op, input, assert_)


unary_ops_bool = [
    torch.bitwise_not,
]


@pytest.mark.parametrize("op", unary_ops_bool)
def test_unary_ops_bool(op):
    torch.manual_seed(42)

    input = torch.randint(2, [1, 2, 10, 200]) > 0

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    unary_op_harness(op, input, assert_)


# Parameterize torch.clamp unittests for different supported overloads
clamp_inputs = [{"min": 0.2, "max": 0.8}, {"min": 0.2}, {"max": 0.8}]


@pytest.mark.parametrize("args", clamp_inputs)
def test_clamp(args):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def op_clamp(x):
        return x.clamp(**args)

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    unary_op_harness(op_clamp, input, assert_)


@pytest.mark.parametrize("args", clamp_inputs)
def test_clamp_(args):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def op_clamp_(x):
        return x.clamp_(**args)

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    unary_op_harness(op_clamp_, input, assert_)


@pytest.mark.parametrize(
    "op",
    [torch.clamp_min, torch.clamp_min_, torch.clamp_max, torch.clamp_max_])
def test_clamp_min(op):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def op_clamp(x):
        return op(x, 0.5)

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    unary_op_harness(op_clamp, input, assert_)


binary_ops_float = [
    torch.add, torch.atan2, torch.div, torch.sub, torch.fmod,
    torch.floor_divide, torch.mul, torch.remainder, torch.true_divide
]


@pytest.mark.parametrize("op", binary_ops_float)
def test_binary_ops_float(op):
    torch.manual_seed(42)

    input1 = torch.randn([1, 2, 5, 1]) * 100.0
    input2 = torch.randn([1, 2, 5, 1]) * 10.0

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    class Model(torch.nn.Module):
        def __init__(self, op):
            super(Model, self).__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x, y)

    binary_op_harness(Model(op), input1, input2, assert_)


binary_ops_basic_element_wise_float = [
    torch.add,
    torch.div,
    torch.sub,
    torch.mul,
]


@pytest.mark.parametrize("op", binary_ops_basic_element_wise_float)
def test_binary_ops_elementwise_edgecases(op):
    torch.manual_seed(42)
    input1 = torch.randn([1, 2, 10, 200])
    input2 = torch.randn([1])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    # Constant on LHS
    class ConstOnLHS(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnLHS, self).__init__()
            self.op = op

        def forward(self, x, _y):
            return self.op(x, 4.0)

    binary_op_harness(ConstOnLHS(op), input1, input2, assert_)

    # Constant on RHS
    class ConstOnRHS(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnRHS, self).__init__()
            self.op = op

        def forward(self, x, _y):
            return self.op(2.5, x)

    binary_op_harness(ConstOnRHS(op), input1, input2, assert_)

    # Constant on LHS wrong type.
    class ConstOnLHSInt(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnLHSInt, self).__init__()
            self.op = op

        def forward(self, x, _y):
            return self.op(x, 4)

    binary_op_harness(ConstOnLHSInt(op), input1, input2, assert_)

    # Constant on RHS wrong type
    class ConstOnRHSInt(torch.nn.Module):
        def __init__(self, op):
            super(ConstOnRHSInt, self).__init__()
            self.op = op

        def forward(self, x, _y):
            return self.op(134, x)

    binary_op_harness(ConstOnRHSInt(op), input1, input2, assert_)


binary_ops_basic_element_wise_bool = [
    torch.add,
    torch.mul,
]


@pytest.mark.parametrize("op", binary_ops_basic_element_wise_bool)
def test_binary_ops_elementwise_bools(op):
    input1 = torch.tensor([False, True, False, True])
    input2 = torch.tensor([False, False, True, True])

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    class BothBools(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x, y)

    binary_op_harness(BothBools(op), input1, input2, assert_)

    class FloatOnLHS(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y):
            x = x.to(torch.float) + 1.0
            return self.op(x, y)

    binary_op_harness(FloatOnLHS(op), input1, input2, assert_)

    class FloatOnRHS(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y):
            y = y.to(torch.float) + 1.0
            return self.op(x, y)

    binary_op_harness(FloatOnRHS(op), input1, input2, assert_)

    class IntOnLHS(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y):
            x = x.to(torch.int) + 1
            return self.op(x, y)

    binary_op_harness(IntOnLHS(op), input1, input2, assert_)

    class IntOnRHS(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y):
            y = y.to(torch.int) + 1
            return self.op(x, y)

    binary_op_harness(IntOnRHS(op), input1, input2, assert_)


@pytest.mark.parametrize("op", [torch.fmod, torch.remainder])
def test_modulo_mixed_sign(op):
    input1 = torch.tensor([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0])
    input2 = torch.tensor([2.1, -3.4, 8.0, -2.1, 3.4, 5.0])

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out,
                                expected=native_out,
                                atol=1e-05,
                                rtol=1e-05,
                                equal_nan=True)

    binary_op_harness(op, input1, input2, assert_)


binary_op_int = [
    # torch.logical_or, torch.logical_xor, , torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor, torch.logical_and,
]

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
def test_reduction_ops_float(op):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

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

    unary_op_harness(op, input, assert_)


@pytest.mark.parametrize("op", reduction_ops_api2)
@pytest.mark.parametrize("dim", range(4))
@pytest.mark.parametrize("keepdim", [False, True])
def test_reduction_ops_float_api2(op, dim, keepdim):
    if op is torch.norm and keepdim and dim == 0:
        # TODO: T36427
        pytest.skip("Test test_reduction_ops_float_api2[True-0-norm]"
                    " is failing and needs investigation.")

    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

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

    unary_op_harness(operation, input, assert_)


@pytest.mark.parametrize("op", [torch.min, torch.max])
@pytest.mark.parametrize("dim", range(3))
@pytest.mark.parametrize("keepdim", [False, True])
def test_minmax_tuple_out(op, dim, keepdim):
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def operation(x):
        return op(x, dim=dim, keepdim=keepdim)

    def assert_(native_out, poptorch_out):
        assert isinstance(native_out, tuple) and isinstance(
            poptorch_out, tuple)
        assert len(native_out) == len(poptorch_out)
        for i, native in enumerate(native_out):
            helpers.assert_allclose(actual=poptorch_out[i], expected=native)

    unary_op_harness(operation, input, assert_)


# Interesting p-values for testing torch.norm(X, p=<>)
norm_pvals = ['fro', float('inf'), float('-inf'), 1, 1.0, 2, 2.0, 3, 3.0]


@pytest.mark.parametrize("p", norm_pvals)
def test_norm_p_values(p):
    torch.manual_seed(42)
    input = torch.randn([1, 2, 10, 200])

    def operation(x):
        return torch.norm(x, p=p)

    def assert_(native_out, poptorch_out):
        helpers.assert_allclose(actual=poptorch_out, expected=native_out)

    unary_op_harness(operation, input, assert_)


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
def test_compare_operations(op):
    torch.manual_seed(42)

    lhs = torch.randn([1, 2, 10, 200])
    rhs = torch.randn([1, 2, 10, 200])

    indices = torch.randint(0, 200, [30])

    # Make a few of the indices equal.
    for i in indices:
        lhs[0][0][0][i] = rhs[0][0][0][i]

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    binary_op_harness(op, lhs, rhs, assert_)

    if op not in (torch.min, torch.max):

        def constant_rhs(x):
            return op(x, 0.34)

        unary_op_harness(constant_rhs, lhs, assert_)


comparison_unity_nan_inf_ops = [
    # torch.isfinite, torch.isinf,  # Not in Onnx
    torch.isnan,
]


@pytest.mark.parametrize("op", comparison_unity_nan_inf_ops)
def test_compare_unity_nan_inf_ops(op):
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

    unary_op_harness(op, input, assert_)


comparison_unity = [torch.max, torch.min]


@pytest.mark.parametrize("op", comparison_unity)
def test_compare_unity_operations(op):
    torch.manual_seed(42)
    input = torch.randn([1, 2, 10, 200])

    def operation(x):
        return op(x)

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    unary_op_harness(operation, input, assert_)


# Support other arguments. TODO(T23319)
def test_topk():
    torch.manual_seed(42)

    input = torch.randn([1, 2, 10, 200])

    def operation(x):
        return torch.topk(x, k=10, dim=-1)

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out[0],
                                expected=native_out.values)
        helpers.assert_allequal(actual=poptorch_out[1],
                                expected=native_out.indices)

    unary_op_harness(operation, input, assert_)


types = [torch.float32, torch.int32]


@pytest.mark.parametrize("ty", types)
def test_constant_arrays(ty):
    torch.manual_seed(42)

    input = torch.randn([10]).to(ty)

    def operation(x):
        constant_tensor = torch.tensor([1, -2, -3, 4, 5, 6, 7, -8, 9, -10],
                                       dtype=ty)
        return torch.sub(x, constant_tensor)

    def assert_(native_out, poptorch_out):
        helpers.assert_allequal(actual=poptorch_out, expected=native_out)

    unary_op_harness(operation, input, assert_)


@pytest.mark.parametrize("ty", types)
def test_big_constant_arrays_sliced(ty):
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

    unary_op_harness(operation, input, assert_)


# Parametrize input tensor shapes for addcdiv to make sure broadcasting works.
broadcastable_shapes = [
    ((3, 1), (3, 1), (3, 1)),
    ((1, 3), (3, 1), (1, 3)),
    ((5, 3), (5, 1), (1, 3)),
    ((1, ), (3, 1), (2, )),
]


@pytest.mark.parametrize("shapes", broadcastable_shapes)
@pytest.mark.parametrize("scale", [0.35, 4.91, 12.0, -0.53, -3.45, -9.0, 0.0])
def test_addcdiv(shapes, scale):
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

    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(t0, t1, t2)

    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


cross_shapes = [(3, 4, 5, 6), (4, 3, 5, 6), (4, 5, 3, 6), (4, 5, 6, 3),
                (6, 3, 3, 5)]


@pytest.mark.parametrize("shape", cross_shapes)
def test_cross_shape(shape):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.cross(x, y)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    torch.manual_seed(42)
    x = torch.randn(shape)
    y = torch.randn(shape)

    native_out = model(x, y)
    poptorch_out = poptorch_model(x, y)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.parametrize("axis", range(0, 4))
def test_cross_axis(axis):
    class Model(torch.nn.Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, x, y):
            return torch.cross(x, y, self.axis)

    model = Model(axis)
    poptorch_model = poptorch.inferenceModel(model)

    torch.manual_seed(42)
    x = torch.randn(3, 3, 3, 3)
    y = torch.randn(3, 3, 3, 3)

    native_out = model(x, y)
    poptorch_out = poptorch_model(x, y)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)
