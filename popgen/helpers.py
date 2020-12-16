# Copyright (c) 2020 Graphcore Ltd. All rights reserved
from popgen import values


# alpha(m, a):
#
# Generate the alpha computation required for operators that have implicit scaling
# Parameters:
#   m - quantity to be scaled
#   a - scaling factor
def alpha(m, a):
    return values.AlphaValue([m, a])


# cint(n)
#
# Generate an integer as a C int literal or variable
# Parameters:
#   n - value to be generated
def cint(n):
    return values.NonTensorConstant('NonTensorInt', n, 'constantToInt')


# clong(n)
#
# Generate an integer as a C long literal or variable
# Parameters
#   n - value to be generated
def clong(n):
    return values.NonTensorConstant('NonTensorLong', n, 'constantToLong')


# clong(l)
#
# Generate a value as a list of C longs
# Parameters
#   l - value to be generated
def clong_list(l):
    return values.NonTensorHelper('ConstantLongList', [l], 'constantToLongVec',
                                  True)


# cfloat(f)
#
# Generate a floating point as a C float literal or variable
# Parameters
#   f - value to be generated
def cfloat(f):
    return values.NonTensorConstant('NonTensorFloat', f, 'constantToFloat')


# cstr(s)
#
# Generate a string as a C string literal or variable
# Parameters
#   s - value to be generated
def cstr(s):
    return values.NonTensorConstant('NonTensorString', s, 'constantToString')


# dimension(a)
# Helper for parameters that are dimensional indices
# Parameters:
#   v - value representing a dimensional index
def dimension(v):
    return values.NonTensorHelper('Dimension', [v], 'handleDimensionParam')


# dimension_list(t)
# Produces a list with the dimensions of a tensor. Needed for some
# reduction operators.
# Parameters:
#   t - input tensor
def dimension_list(t):
    return values.NonTensorHelper('DimensionList', [t],
                                  "reduceHelperDimensionCreator")


# output_shape(index = 0)
#
# Generate a tensor shape for the output value.
# Parameters
#   index - index of output (default: 0)
def output_shape(idx=0):
    return tensor_shape(values.OutputValue(idx))


# tensor_list(l)
#
# Generate a list of tensors
# Parameters
#   l - value to be generated
def tensor_list(l):
    return values.Helper('TensorList', [l], "handleTensorList", True)


# tensor_long(t)
#
# Change the scalar type of a tensor to Long
# Parameters
#   s - input tensor
def tensor_long(t):
    return values.CastInPlace("inplace_cast<long>", [t],
                              'at::ScalarType::Long')


# tensor_shape(t)
#
# Generate the shape of a tensor as a C++ vector of ints
# Parameters
#   t - input tensor
def tensor_shape(t):
    return values.NonTensorHelper('TensorShape', [t], "shapeFromTensor")
