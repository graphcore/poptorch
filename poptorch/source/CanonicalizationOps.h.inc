// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// aten::dropout(Tensor input, float p, bool train) -> Tensor
OP_CONVERTOR(aten::dropout, NONE, createDropout,
             {PARAM(0)} COMMA 1 COMMA HANDLE(1, float))
OP_CONVERTOR(aten::dropout_, NONE, createDropout,
             {PARAM(0)} COMMA 1 COMMA HANDLE(1, float))

// aten::t(Tensor self) -> Tensor
OP_CONVERTOR(aten::t, NONE, createTranspose, {PARAM(0)} COMMA{})

// aten::relu(Tensor self) -> Tensor
OP_CONVERTOR(aten::relu, NONE, createRelu, {PARAM(0)})

// aten::relu_(Tensor self) -> Tensor
OP_CONVERTOR(aten::relu_, NONE, createRelu, {PARAM(0)})

// aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor
OP_CONVERTOR(aten::sub, ALPHA(PARAM_OR_CONSTANT_ANY_TYPE(1), PARAM(2)), createSub,
             {PARAM_OR_CONSTANT_ANY_TYPE(0) COMMA alphaValue})

// aten::mul(Tensor self, Tensor other)
OP_CONVERTOR(aten::mul, NONE, createMul,
             {PARAM_OR_CONSTANT_ANY_TYPE(0) COMMA PARAM_OR_CONSTANT_ANY_TYPE(1)})

// aten::div(Tensor self, Tensor other)
OP_CONVERTOR(aten::div, NONE, createDiv,
             {PARAM_OR_CONSTANT_ANY_TYPE(0) COMMA PARAM_OR_CONSTANT_ANY_TYPE(1)})

// aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor
OP_CONVERTOR(aten::add, ALPHA(PARAM_OR_CONSTANT_ANY_TYPE(1), PARAM(2)), createAdd,
             {PARAM_OR_CONSTANT_ANY_TYPE(0) COMMA alphaValue})

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar
// alpha) -> Tensor
OP_CONVERTOR(aten::addmm, NONE, createGemm,
             {PARAM(1) COMMA PARAM(2) COMMA PARAM(0)} COMMA HANDLE(3, int)
                 COMMA HANDLE(4, int) COMMA 0 COMMA 0)

// aten::tanh(Tensor self) -> Tensor
OP_CONVERTOR(aten::tanh, NONE, createTanh, {PARAM(0)})

// aten::gelu(Tensor self) -> Tensor
OP_CONVERTOR(aten::gelu, NONE, createGelu, {PARAM(0)})

// aten::matmul(Tensor self, Tensor other) -> Tensor
OP_CONVERTOR(aten::matmul, NONE, createMatmul, {PARAM(0) COMMA PARAM(1)})


// aten::sigmoid(Tensor t) -> Tensor
OP_CONVERTOR(aten::sigmoid, NONE, createSigmoid, {PARAM(0)})


// Pads are all in the form of:
// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
// Only constant has the |value| the multiple dimension of inputs vary by the length of pads.

OP_CONVERTOR(aten::constant_pad_nd, NONE, createConstantPad,
             PARAM(0) COMMA HANDLE_LIST(1, std::int64_t) COMMA HANDLE(2, float))

OP_CONVERTOR(aten::reflection_pad1d, NONE, createReflectionPad,
             PARAM(0) COMMA HANDLE_LIST(1, std::int64_t))
OP_CONVERTOR(aten::reflection_pad2d, NONE, createReflectionPad,
             PARAM(0) COMMA HANDLE_LIST(1, std::int64_t))


OP_CONVERTOR(aten::replication_pad1d, NONE, createEdgePad,
             PARAM(0) COMMA HANDLE_LIST(1, std::int64_t))
OP_CONVERTOR(aten::replication_pad2d, NONE, createEdgePad,
             PARAM(0) COMMA HANDLE_LIST(1, std::int64_t))
OP_CONVERTOR(aten::replication_pad3d, NONE, createEdgePad,
             PARAM(0) COMMA HANDLE_LIST(1, std::int64_t))

// aten::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias,
// float eps, bool cudnn_enabled) -> Tensor
OP_CONVERTOR(aten::group_norm, NONE, createGroupnormalization,
             {PARAM(0) COMMA PARAM(2) COMMA PARAM(3)} COMMA HANDLE(1,
                                                                   std::int64_t)
                 COMMA HANDLE(4, float))

// aten::cat(Tensor[] tensors, int dim=0) -> Tensor
OP_CONVERTOR(aten::cat, NONE, createConcat,
             HANDLE_TENSOR_LIST(0) COMMA HANDLE(1, std::int64_t))


OP_CONVERTOR(aten::pow, NONE, createPow, {PARAM(0) COMMA PARAM_OR_CONSTANT(1, float)})

// Math unary operations.
OP_CONVERTOR(aten::abs, NONE, createAbs, {PARAM(0)})
OP_CONVERTOR(aten::acos, NONE, createAcos, {PARAM(0)})
OP_CONVERTOR(aten::asin, NONE, createAsin, {PARAM(0)})
OP_CONVERTOR(aten::atan, NONE, createAtan, {PARAM(0)})
OP_CONVERTOR(aten::ceil, NONE, createCeil, {PARAM(0)})
OP_CONVERTOR(aten::cos, NONE, createCos, {PARAM(0)})
OP_CONVERTOR(aten::cosh, NONE, createCosh, {PARAM(0)})
OP_CONVERTOR(aten::erf, NONE, createErf, {PARAM(0)})
OP_CONVERTOR(aten::exp, NONE, createExp, {PARAM(0)})
OP_CONVERTOR(aten::floor, NONE, createFloor, {PARAM(0)})
OP_CONVERTOR(aten::log, NONE, createLog, {PARAM(0)})
OP_CONVERTOR(aten::neg, NONE, createNeg, {PARAM(0)})
OP_CONVERTOR(aten::reciprocal, NONE, createReciprocal, {PARAM(0)})
OP_CONVERTOR(aten::sin, NONE, createSin, {PARAM(0)})
OP_CONVERTOR(aten::sinh, NONE, createSinh, {PARAM(0)})
OP_CONVERTOR(aten::sign, NONE, createSign, {PARAM(0)})
OP_CONVERTOR(aten::sqrt, NONE, createSqrt, {PARAM(0)})
OP_CONVERTOR(aten::square, NONE, createMul, {PARAM(0) COMMA PARAM(0)})
OP_CONVERTOR(aten::tan, NONE, createTan, {PARAM(0)})
OP_CONVERTOR(aten::tanh, NONE, createTanh, {PARAM(0)})
OP_CONVERTOR(aten::logical_not, NONE, createLogical_not, {PARAM(0)})


// Activations
OP_CONVERTOR(aten::selu, NONE, createSelu, {PARAM(0)} COMMA 1.6732632423543772848170429916717f COMMA 1.0507009873554804934193349852946f)
OP_CONVERTOR(aten::selu_, NONE, createSelu, {PARAM(0)} COMMA 1.6732632423543772848170429916717f COMMA 1.0507009873554804934193349852946f)
// Re-enable after release (in popart master)
// OP_CONVERTOR(aten::prelu, NONE, Create_prelu, {PARAM(0) COMMA PARAM(1)})
OP_CONVERTOR(aten::elu, NONE, createElu, {PARAM(0)} COMMA HANDLE(1, float))
OP_CONVERTOR(aten::elu_, NONE, createElu, {PARAM(0)} COMMA HANDLE(1, float))
OP_CONVERTOR(aten::leaky_relu, NONE, createLeakyrelu, {PARAM(0)} COMMA HANDLE(1, float))
OP_CONVERTOR(aten::leaky_relu_, NONE, createLeakyrelu, {PARAM(0)} COMMA HANDLE(1, float))


// Comparison ops.
OP_CONVERTOR(aten::eq, NONE, createEqual, {PARAM(0) COMMA PARAM_OR_CONSTANT_ANY_TYPE_NO_CAST(1)})
OP_CONVERTOR(aten::gt, NONE, createGreater, {PARAM(0) COMMA PARAM_OR_CONSTANT_ANY_TYPE_NO_CAST(1)})
OP_CONVERTOR(aten::lt, NONE, createLess, {PARAM(0) COMMA PARAM_OR_CONSTANT_ANY_TYPE_NO_CAST(1)})


// Max/min between two tensors.
OP_CONVERTOR(aten::max and NUM_INPUTS_EQUALS(2), NONE, createMax, {PARAM(0) COMMA PARAM(1)})
OP_CONVERTOR(aten::min and NUM_INPUTS_EQUALS(2), NONE, createMin, {PARAM(0) COMMA PARAM(1)})

// Max/Min reduce. Takes one tensor and reduces it.
OP_CONVERTOR(aten::max and NUM_INPUTS_EQUALS(1), NONE, createReducemax, {PARAM(0)} COMMA DIMENISON_LENGTH_LIST(0) COMMA 0)
OP_CONVERTOR(aten::min and NUM_INPUTS_EQUALS(1), NONE, createReducemin, {PARAM(0)} COMMA DIMENISON_LENGTH_LIST(0) COMMA 0)

// Same as above with dim and keepdim args.
// Unsupported due to PopART/Onnx not returning the indices as well. TODO(T23333)
//OP_CONVERTOR(aten::max and NUM_INPUTS_EQUALS(3), NONE, createReducemax, {PARAM(0)} COMMA {HANDLE_DIM(1)} COMMA HANDLE(2, int64_t))
//OP_CONVERTOR(aten::min and NUM_INPUTS_EQUALS(3), NONE, createReducemin, {PARAM(0)} COMMA {HANDLE_DIM(1)} COMMA HANDLE(2, int64_t))
OP_CONVERTOR(aten::topk, NONE, createTopk, {PARAM(0)} COMMA HANDLE(1, int64_t) COMMA  HANDLE_DIM(2))

// Opset 10
//OP_CONVERTOR(aten::isinf, NONE, createIsinf, {PARAM(0)} COMMA 1 COMMA 1)
OP_CONVERTOR(aten::isnan, NONE, createIsnan, {PARAM(0)})
