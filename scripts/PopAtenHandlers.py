#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import math
from popgen.api import convert, expand, forward, generate, simplify
from popgen.helpers import as_ir, alpha, cfloat, cint, clong, clong_list, \
                           cstr, dimension, empty_initializer, output_shape, \
                           output_type, reduction, tensor_list, tensor_long, \
                           tensor_shape, tensor_type
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
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil", "cos",
    "cosh", "detach", "erf", "exp", "expm1", "floor", "gelu", "isnan", "log",
    "log1p", "logical_not", "neg", "reciprocal", "relu", "round", "sigmoid",
    "sin", "sinh", "sign", "sqrt", "tan", "tanh"
]

for oper in opers:
    convert(oper, 1)

convert("t", 1, "transpose")

expand("erfc", lambda x: 1. - op.erf(x))
expand("log2", lambda x: op.log(x) / math.log(2))
expand("log10", lambda x: op.log(x) / math.log(10))
expand("log_sigmoid", lambda x: op.log(op.sigmoid(x)))
expand(
    "rand", lambda x: op.randomUniform(x, output_shape(), cfloat(1.), cfloat(
        0.), output_type()))
expand(
    "randn", lambda: op.randomNormal(empty_initializer(), output_shape(),
                                     cfloat(0.), cfloat(1.), output_type()))
expand("rsqrt", lambda x: 1. / op.sqrt(x))
expand("selu", lambda x: op.selu(x, cfloat(selu_alpha), cfloat(selu_lambda)))
expand("silu", lambda x: x * op.sigmoid(x))
expand("square", lambda x: x * x)

# binary operators
opers = ["atan2", "div", "fmod", "pow", "prelu", "remainder"]

for oper in opers:
    convert(oper, 2)

convert("eq", 2, "equal")
convert("gt", 2, "greater")
convert("lt", 2, "less")

convert("logical_and", 2)
convert("logical_or", 2)

expand("cat", lambda x, y: op.concat(tensor_list(x), clong(y)))
expand("elu", lambda x, y: op.elu(x, cfloat(y)))
expand("full_like", lambda x, y: op.expand(y, as_ir(tensor_shape(x))))
expand("ge", lambda x, y: x >= y)
expand("le", lambda x, y: x <= y)
expand("leaky_relu", lambda x, y: op.leakyrelu(x, cfloat(y)))
expand("ne", lambda x, y: x != y)
expand("pixel_shuffle", lambda x, y: op.depthtospace(x, clong(y), cstr("CRD")))
expand("reflection_pad1d", lambda x, y: op.reflectionPad(x, clong_list(y)))
expand("replication_pad1d", lambda x, y: op.edgePad(x, clong_list(y)))
expand("rsub", lambda x, y: y - x)


def celu_handler(x, a):
    val = a * (op.exp(x / a) - 1.)
    return op.max(x, 0.) + op.min(0., val)


def full_handler(x, y):
    r = op.expand(y, as_ir(clong_list(x)))
    return op.cast(r, output_type())


def hardshrink_handler(x, l):
    return op.where(op.abs(x) > op.abs(l), x, 0.)


def softshrink_handler(x, l):
    r = op.where(x > l, x - l, 0.)
    return op.where(x < -l, x + l, r)


forward("reflection_pad2d", "reflection_pad1d")
forward("replication_pad2d", "replication_pad1d")
forward("replication_pad3d", "replication_pad1d")

# ternary operators
convert("masked_fill", 3, "where", [1, 2, 0])
convert("where", 3)

expand("constant_pad_nd", lambda x, l, c: op.constantPad(
    x, clong_list(l), cfloat(c)))
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
expand("threshold", lambda x, threshold, val: op.where(x > threshold, x, val))

expand("index_select", lambda x, d, i: op.gather(x, i,
                                                 dimension(d, tensor_type(x))))


def softplus_handler(x, b, threshold):
    condition = x * b > threshold
    softplus = 1. / b * op.log(1. + op.exp(b * x))
    return op.where(condition, x, softplus)


# loss handlers
def hinge_embedding_loss_handler(x, y, delta, red):
    red = reduction(clong(red))
    loss = op.where(y.equal(1.), x, 0.)
    loss = op.where(y.equal(-1.), op.max(0., delta - x), loss)
    return op.identityloss(loss, red)


def l1_loss_handler(x, y, red):
    red = reduction(clong(red))
    loss = op.l1loss(x - y, cfloat(1.), red)
    return op.identityloss(loss, cint(2))


def margin_ranking_loss_handler(x1, x2, y, margin, red):
    red = reduction(clong(red))
    loss = op.max(-y * (x1 - x2) + margin, 0.)
    return op.identityloss(loss, red)


def mse_loss_handler(x, y, red):
    red = reduction(clong(red))
    loss = (x - y) * (x - y)
    return op.identityloss(loss, red)


def smooth_l1_loss_handler(x, y, red, beta):
    red = reduction(clong(red))
    delta = op.abs(x - y)
    loss = op.where(delta < beta, 0.5 * delta * delta / beta,
                    delta - 0.5 * beta)
    return op.identityloss(loss, red)


def soft_margin_loss_handler(x, y, red):
    red = reduction(clong(red))
    loss = op.log(1. + op.exp(-y * x))
    return op.identityloss(loss, red)


# everything else
expand(
    "addmm", lambda x, y, z, c1, c2: op.gemm(y, z, x, cfloat(c1), cfloat(c2),
                                             clong(0), clong(0)))

generate(script, "c10::aten", output_dir + "/AtenHandlers.gen.cpp", globals())
