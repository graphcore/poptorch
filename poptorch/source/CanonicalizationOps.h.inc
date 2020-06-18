// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// aten::dropout(Tensor input, float p, bool train) -> Tensor
OP_CONVERTOR(aten::dropout, NONE, Create_dropout,
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

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor
OP_CONVERTOR(aten::addmm, NONE, Create_gemm,
             {PARAM(1) COMMA PARAM(2) COMMA PARAM(0)} COMMA HANDLE(3, int)
                  COMMA HANDLE(4, int) COMMA 0 COMMA 0)

// aten::tanh(Tensor self) -> Tensor
OP_CONVERTOR(aten::tanh, NONE, Create_tanh, {PARAM(0)})

// aten::gelu(Tensor self) -> Tensor
OP_CONVERTOR(aten::gelu, NONE, Create_gelu, {PARAM(0)})

// aten::matmul(Tensor self, Tensor other) -> Tensor
OP_CONVERTOR(aten::matmul, NONE, Create_matmul, {PARAM(0) COMMA PARAM(1)})

// aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, bool cudnn_enable) -> Tensor
OP_CONVERTOR(aten::layer_norm, NONE, Create_groupnormalization, {PARAM(0) COMMA PARAM(2) COMMA PARAM(3)} COMMA 1 COMMA HANDLE(4, float))

// "aten::mean(Tensor self, *, int? dtype) -> Tensor"
OP_CONVERTOR(aten::mean, NONE, Create_mean, {PARAM(0)})
