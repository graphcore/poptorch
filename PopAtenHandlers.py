#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from popgen.api import convert, expand, forward, generate, simplify
from popgen.helpers import as_ir, alpha, cfloat, cint, clong, clong_list, \
                           cstr, dimension, dimension_list, empty_initializer, \
                           output_shape, output_type, tensor_list, \
                           tensor_long, tensor_shape, tensor_type
from popgen.operatorfactory import op

script = "PopAtenHandlers.py"
output_dir = "poptorch/source/popart_canonicalization"

selu_alpha = 1.6732632423543772848170429916717
selu_lambda = 1.0507009873554804934193349852946

# simplification rules
simplify("expm1", lambda x: op.exp(x) - 1.)
simplify("log1p", lambda x: op.log(1. + x))
simplify("reciprocal", lambda x: 1. / x)
simplify("div", lambda x, y: 1. / y * x)

# unary operators
opers = [
    "abs", "acos", "asin", "atan", "ceil", "cos", "cosh", "detach", "erf",
    "exp", "expm1", "floor", "gelu", "isnan", "log", "logical_not", "neg",
    "reciprocal", "relu", "round", "sigmoid", "sin", "sinh", "sign", "sqrt",
    "tan", "tanh"
]

for oper in opers:
    convert(oper, 1)

convert("t", 1, "transpose")

expand("frobenius_norm", lambda x: op.reducel2(x, dimension_list(x), clong(0)))
expand("max", lambda x: op.reducemax(x, dimension_list(x), clong(0)))
expand("min", lambda x: op.reducemin(x, dimension_list(x), clong(0)))
expand(
    "rand", lambda x: op.randomUniform(x, output_shape(), cfloat(1.), cfloat(
        0.), output_type()))
expand(
    "randn", lambda: op.randomNormal(empty_initializer(), output_shape(),
                                     cfloat(0.), cfloat(1.), output_type()))
expand("rsqrt", lambda x: 1. / op.sqrt(x))
expand("selu", lambda x: op.selu(x, cfloat(selu_alpha), cfloat(selu_lambda)))
expand("square", lambda x: x * x)

forward("relu_", "relu")
forward("selu_", "selu")

# binary operators
opers = ["atan2", "div", "fmod", "max", "min", "pow", "prelu", "remainder"]

for oper in opers:
    convert(oper, 2)

convert("eq", 2, "equal")
convert("gt", 2, "greater")
convert("lt", 2, "less")

expand("cat", lambda x, y: op.concat(tensor_list(x), clong(y)))
expand("dropout", lambda x, y: op.dropout(x, cint(1), cfloat(y)))
expand("elu", lambda x, y: op.elu(x, cfloat(y)))
expand("full_like", lambda x, y: op.expand(y, as_ir(tensor_shape(x))))

expand("leaky_relu", lambda x, y: op.leakyrelu(x, cfloat(y)))
expand("pixel_shuffle", lambda x, y: op.depthtospace(x, clong(y), cstr("CRD")))
expand("reflection_pad1d", lambda x, y: op.reflectionPad(x, clong_list(y)))
expand("replication_pad1d", lambda x, y: op.edgePad(x, clong_list(y)))


def celu_handler(x, a):
    val = a * (op.exp(x / a) - 1.)
    return op.max(x, 0.) + op.min(0., val)


def full_handler(x, y):
    r = op.expand(y, as_ir(clong_list(x)))
    return op.cast(r, output_type())


forward("dropout_", "dropout")
forward("elu_", "elu")
forward("leaky_relu_", "leaky_relu")
forward("prelu_", "prelu")
forward("reflection_pad2d", "reflection_pad1d")
forward("replication_pad2d", "replication_pad1d")
forward("replication_pad3d", "replication_pad1d")

# ternary operators
convert("masked_fill", 3, "where", [1, 2, 0])
convert("where", 3)

expand("clamp", lambda x, y, z: op.clip(x, cfloat(z), cfloat(y)))
expand("constant_pad_nd", lambda x, l, c: op.constantPad(
    x, clong_list(l), cfloat(c)))
expand(
    "frobenius_norm", lambda x, l, c: op.reducel2(
        x, dimension_list(x, clong_list(l)), clong(c)))
expand("hardtanh", lambda x, a, b: op.clip(x, cfloat(b), cfloat(a)))
expand(
    "normal_", lambda x, c1, c2: op.randomNormal(x, tensor_shape(x), cfloat(
        c1), cfloat(c2)))
expand("sub", lambda x, y, a: op.sub(x, alpha(y, a)))
expand(
    "uniform_", lambda x, a, b: op.randomUniform(x, tensor_shape(x), cfloat(b),
                                                 cfloat(a)))
expand(
    "topk", lambda x, c, l: op.topk(x, tensor_long(c),
                                    dimension(l, tensor_type(x))))


def softplus_handler(x, b, threshold):
    condition = x * b > threshold
    softplus = 1. / b * op.log(1. + op.exp(b * x))
    return op.where(condition, x, softplus)


forward("clamp_", "clamp")
forward("hardtanh_", "hardtanh")
forward("masked_fill_", "masked_fill")
forward("where_", "where")

# everything else
expand(
    "addmm", lambda x, y, z, c1, c2: op.gemm(y, z, x, cfloat(c1), cfloat(c2),
                                             clong(0), clong(0)))

generate(script, "c10::aten", output_dir + "/AtenHandlers.gen.cpp", globals())
