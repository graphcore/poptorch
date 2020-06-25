// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// aten::dropout(Tensor input, float p, bool train) -> Tensor
OP_CONVERTOR(aten::dropout, NONE, Create_dropout,
             {PARAM(0)} COMMA 1 COMMA HANDLE(1, float))
OP_CONVERTOR(aten::dropout_, NONE, Create_dropout,
             {PARAM(0)} COMMA 1 COMMA HANDLE(1, float))

// aten::t(Tensor self) -> Tensor
OP_CONVERTOR(aten::t, NONE, Create_transpose, {PARAM(0)} COMMA{})

// aten::relu(Tensor self) -> Tensor
OP_CONVERTOR(aten::relu, NONE, Create_relu, {PARAM(0)})

// aten::relu_(Tensor self) -> Tensor
OP_CONVERTOR(aten::relu_, NONE, Create_relu, {PARAM(0)})

// aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor
OP_CONVERTOR(aten::sub, ALPHA(PARAM(1), PARAM(2)), Create_sub,
             {PARAM(0) COMMA alphaValue})

// aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor
OP_CONVERTOR(aten::add, ALPHA(PARAM(1), PARAM(2)), Create_add,
             {PARAM(0) COMMA alphaValue})

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar
// alpha) -> Tensor
OP_CONVERTOR(aten::addmm, NONE, Create_gemm,
             {PARAM(1) COMMA PARAM(2) COMMA PARAM(0)} COMMA HANDLE(3, int)
                 COMMA HANDLE(4, int) COMMA 0 COMMA 0)

// aten::tanh(Tensor self) -> Tensor
OP_CONVERTOR(aten::tanh, NONE, Create_tanh, {PARAM(0)})

// aten::gelu(Tensor self) -> Tensor
OP_CONVERTOR(aten::gelu, NONE, Create_gelu, {PARAM(0)})

// aten::matmul(Tensor self, Tensor other) -> Tensor
OP_CONVERTOR(aten::matmul, NONE, Create_matmul, {PARAM(0) COMMA PARAM(1)})


// aten::sigmoid(Tensor t) -> Tensor
OP_CONVERTOR(aten::sigmoid, NONE, Create_sigmoid, {PARAM(0)})

// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
OP_CONVERTOR(aten::constant_pad_nd, NONE, Create_ConstantPad,
             PARAM(0) COMMA HANDLE_LIST(1, std::int64_t) COMMA HANDLE(2, float))

// aten::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias,
// float eps, bool cudnn_enabled) -> Tensor
OP_CONVERTOR(aten::group_norm, NONE, Create_groupnormalization,
             {PARAM(0) COMMA PARAM(2) COMMA PARAM(3)} COMMA HANDLE(1,
                                                                   std::int64_t)
                 COMMA HANDLE(4, float))

// aten::cat(Tensor[] tensors, int dim=0) -> Tensor
OP_CONVERTOR(aten::cat, NONE, Create_concat,
             HANDLE_TENSOR_LIST(0) COMMA HANDLE(1, std::int64_t))


OP_CONVERTOR(aten::pow, NONE, Create_pow, {PARAM(0) COMMA PARAM_OR_CONSTANT(1, float)})

// Math unary operations.
OP_CONVERTOR(aten::abs, NONE, Create_abs, {PARAM(0)})
OP_CONVERTOR(aten::acos, NONE, Create_acos, {PARAM(0)})
OP_CONVERTOR(aten::asin, NONE, Create_asin, {PARAM(0)})
OP_CONVERTOR(aten::atan, NONE, Create_atan, {PARAM(0)})
OP_CONVERTOR(aten::ceil, NONE, Create_ceil, {PARAM(0)})
OP_CONVERTOR(aten::cos, NONE, Create_cos, {PARAM(0)})
OP_CONVERTOR(aten::cosh, NONE, Create_cosh, {PARAM(0)})
OP_CONVERTOR(aten::erf, NONE, Create_erf, {PARAM(0)})
OP_CONVERTOR(aten::exp, NONE, Create_exp, {PARAM(0)})
OP_CONVERTOR(aten::floor, NONE, Create_floor, {PARAM(0)})
OP_CONVERTOR(aten::log, NONE, Create_log, {PARAM(0)})
OP_CONVERTOR(aten::neg, NONE, Create_neg, {PARAM(0)})
OP_CONVERTOR(aten::reciprocal, NONE, Create_reciprocal, {PARAM(0)})
OP_CONVERTOR(aten::sin, NONE, Create_sin, {PARAM(0)})
OP_CONVERTOR(aten::sinh, NONE, Create_sinh, {PARAM(0)})
OP_CONVERTOR(aten::sign, NONE, Create_sign, {PARAM(0)})
OP_CONVERTOR(aten::sqrt, NONE, Create_sqrt, {PARAM(0)})
OP_CONVERTOR(aten::square, NONE, Create_mul, {PARAM(0) COMMA PARAM(0)})
OP_CONVERTOR(aten::tan, NONE, Create_tan, {PARAM(0)})
OP_CONVERTOR(aten::tanh, NONE, Create_tanh, {PARAM(0)})
OP_CONVERTOR(aten::logical_not, NONE, Create_logical_not, {PARAM(0)})
