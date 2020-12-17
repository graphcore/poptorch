#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from popgen.api import convert, expand, forward, generate, op, simplify
from popgen.helpers import alpha, cfloat, cint, clong, clong_list, \
                           dimension_list, tensor_list, tensor_shape

script = "PopAtenHandlers.py"
output_dir = "poptorch/source/popart_canonicalization"

selu_alpha = 1.6732632423543772848170429916717
selu_lambda = 1.0507009873554804934193349852946

# simplification rules
simplify("expm1", lambda x: op.exp(x) - 1.)
simplify("log1p", lambda x: op.log(1. + x))
simplify("reciprocal", lambda x: 1. / x)
simplify("div", lambda x, y: 1. / y * x)

expand(
    "addmm", lambda x, y, z, c1, c2: op.gemm(y, z, x, cfloat(c1), cfloat(c2),
                                             cint(0), cint(0)))

expand("cat", lambda x, y: op.concat(tensor_list(x), clong(y)))

expand("clamp", lambda x, y, z: op.clip(x, cfloat(z), cfloat(y)))

expand("dropout", lambda x, y: op.dropout(x, cint(1), cfloat(y)))

convert("relu", 1)
forward("relu_", "relu")

expand("sub", lambda x, y, a: op.sub(x, alpha(y, a)))

# simple conversion of a binary operand
convert("atan2", 2)
convert("pow", 2)

# swizzles for inputs
convert("square", 1, "mul", [0, 0])
convert("masked_fill", 3, "where", [1, 2, 0])
forward("masked_fill_", "masked_fill")

convert("where", 3, "where", [0, 1, 2])
forward("where_", "where")

# converting to different operators based on arity
expand("min", lambda x: op.reducemin(x, dimension_list(x), cint(0)))
convert("min", 2)

expand(
    "normal_", lambda x, c1, c2: op.randomNormal(x, tensor_shape(x), cfloat(
        c1), cfloat(c2)))

expand("selu", lambda x: op.selu(x, cfloat(selu_alpha), cfloat(selu_lambda)))
forward("selu_", "selu")


# examples of using a python function. these are added as handlers implicitly
# based on naming. convetion is: <operator_name>_handler
def celu_handler(x, a):
    val = a * (op.exp(x / a) - 1.)
    return op.max(x, 0.) + op.min(0., val)


def softplus_handler(x, beta, threshold):
    condition = x * beta > threshold
    softplus = 1. / beta * op.log(1. + op.exp(beta * x))
    return op.where(condition, x, softplus)


expand("rsqrt", lambda x: 1. / op.sqrt(x))

# emit C++ code
generate(script, "c10::aten", output_dir + "/AtenHandlers.gen.cpp", globals())
