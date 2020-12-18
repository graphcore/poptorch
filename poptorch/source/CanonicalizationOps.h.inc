// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// aten::dropout(Tensor input, float p, bool train) -> Tensor
OP_CONVERTOR(aten::dropout_, NONE, createDropout,
             {PARAM(0)} COMMA 1 COMMA CONSTANT_TO_FLOAT(1))

// aten::t(Tensor self) -> Tensor
OP_CONVERTOR(aten::t, NONE, createTranspose, {PARAM(0)} COMMA{})

// aten::div(Tensor self, Tensor other)
OP_CONVERTOR(aten::div, NONE, createDiv,
             {PARAM(0) COMMA PARAM(1)})

// aten::tanh(Tensor self) -> Tensor
OP_CONVERTOR(aten::tanh, NONE, createTanh, {PARAM(0)})

// aten::gelu(Tensor self) -> Tensor
OP_CONVERTOR(aten::gelu, NONE, createGelu, {PARAM(0)})

// aten::bmm(Tensor self, Tensor other) -> Tensor
OP_CONVERTOR(aten::bmm, NONE, createMatmul, {PARAM(0) COMMA PARAM(1)})

// aten::sigmoid(Tensor t) -> Tensor
OP_CONVERTOR(aten::sigmoid, NONE, createSigmoid, {PARAM(0)})

// aten::detach(Tensor self) -> Tensor
OP_CONVERTOR(aten::detach, NONE, createDetach, {PARAM(0)})

// Pads are all in the form of:
// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
// Only constant has the |value| the multiple dimension of inputs vary by the length of pads.

OP_CONVERTOR(aten::constant_pad_nd, NONE, createConstantPad,
             PARAM(0) COMMA HANDLE_LIST(1) COMMA CONSTANT_TO_FLOAT(2))

OP_CONVERTOR(aten::reflection_pad1d, NONE, createReflectionPad,
             PARAM(0) COMMA HANDLE_LIST(1))
OP_CONVERTOR(aten::reflection_pad2d, NONE, createReflectionPad,
             PARAM(0) COMMA HANDLE_LIST(1))


OP_CONVERTOR(aten::replication_pad1d, NONE, createEdgePad,
             PARAM(0) COMMA HANDLE_LIST(1))
OP_CONVERTOR(aten::replication_pad2d, NONE, createEdgePad,
             PARAM(0) COMMA HANDLE_LIST(1))
OP_CONVERTOR(aten::replication_pad3d, NONE, createEdgePad,
             PARAM(0) COMMA HANDLE_LIST(1))


// Math unary operations.
OP_CONVERTOR(aten::abs, NONE, createAbs, {PARAM(0)})
OP_CONVERTOR(aten::acos, NONE, createAcos, {PARAM(0)})
OP_CONVERTOR(aten::asin, NONE, createAsin, {PARAM(0)})
OP_CONVERTOR(aten::atan, NONE, createAtan, {PARAM(0)})
OP_CONVERTOR(aten::ceil, NONE, createCeil, {PARAM(0)})
OP_CONVERTOR(aten::clamp_, NONE, createClip, {PARAM(0)} COMMA CONSTANT_TO_FLOAT(2) COMMA CONSTANT_TO_FLOAT(1))
OP_CONVERTOR(aten::cos, NONE, createCos, {PARAM(0)})
OP_CONVERTOR(aten::cosh, NONE, createCosh, {PARAM(0)})
OP_CONVERTOR(aten::erf, NONE, createErf, {PARAM(0)})
OP_CONVERTOR(aten::exp, NONE, createExp, {PARAM(0)})
OP_CONVERTOR(aten::floor, NONE, createFloor, {PARAM(0)})
OP_CONVERTOR(aten::log, NONE, createLog, {PARAM(0)})
OP_CONVERTOR(aten::neg, NONE, createNeg, {PARAM(0)})
OP_CONVERTOR(aten::reciprocal, NONE, createReciprocal, {PARAM(0)})
OP_CONVERTOR(aten::round, NONE, createRound, {PARAM(0)})
OP_CONVERTOR(aten::sin, NONE, createSin, {PARAM(0)})
OP_CONVERTOR(aten::sinh, NONE, createSinh, {PARAM(0)})
OP_CONVERTOR(aten::sign, NONE, createSign, {PARAM(0)})
OP_CONVERTOR(aten::sqrt, NONE, createSqrt, {PARAM(0)})
OP_CONVERTOR(aten::tan, NONE, createTan, {PARAM(0)})
OP_CONVERTOR(aten::tanh, NONE, createTanh, {PARAM(0)})
OP_CONVERTOR(aten::logical_not, NONE, createLogical_not, {PARAM(0)})
OP_CONVERTOR(aten::expm1, NONE, createExpm1, {PARAM(0)})

// Activations
OP_CONVERTOR(aten::prelu, NONE, createPrelu, {PARAM(0) COMMA PARAM(1)})
OP_CONVERTOR(aten::prelu_, NONE, createPrelu, {PARAM(0) COMMA PARAM(1)})

OP_CONVERTOR(aten::elu, NONE, createElu, {PARAM(0)} COMMA CONSTANT_TO_FLOAT(1))
OP_CONVERTOR(aten::elu_, NONE, createElu, {PARAM(0)} COMMA CONSTANT_TO_FLOAT(1))
OP_CONVERTOR(aten::leaky_relu, NONE, createLeakyrelu, {PARAM(0)} COMMA CONSTANT_TO_FLOAT(1))
OP_CONVERTOR(aten::leaky_relu_, NONE, createLeakyrelu, {PARAM(0)} COMMA CONSTANT_TO_FLOAT(1))
OP_CONVERTOR(aten::hardtanh, NONE, createClip, {PARAM(0)} COMMA CONSTANT_TO_FLOAT(2) COMMA CONSTANT_TO_FLOAT(1))
OP_CONVERTOR(aten::hardtanh_, NONE, createClip, {PARAM(0)} COMMA CONSTANT_TO_FLOAT(2) COMMA CONSTANT_TO_FLOAT(1))

// Comparison ops.
OP_CONVERTOR(aten::eq, NONE, createEqual, {PARAM(0) COMMA PARAM(1)})
OP_CONVERTOR(aten::gt, NONE, createGreater, {PARAM(0) COMMA PARAM(1)})
OP_CONVERTOR(aten::lt, NONE, createLess, {PARAM(0) COMMA PARAM(1)})


// Max/min between two tensors.
OP_CONVERTOR(aten::max and NUM_INPUTS_EQUALS(2), NONE, createMax, {PARAM(0) COMMA PARAM(1)})

// Max/Min reduce. Takes one tensor and reduces it.
OP_CONVERTOR(aten::max and NUM_INPUTS_EQUALS(1), NONE, createReducemax, {PARAM(0)} COMMA DIMENSION_LENGTH_LIST(0) COMMA 0)

OP_CONVERTOR(aten::topk, NONE, createTopk, {PARAM(0) COMMA CONSTANT_TO_LONG_CONSTANT(1)} COMMA  HANDLE_DIM(2))
// Opset 10
//OP_CONVERTOR(aten::isinf, NONE, createIsinf, {PARAM(0)} COMMA 1 COMMA 1)
OP_CONVERTOR(aten::isnan, NONE, createIsnan, {PARAM(0)})
OP_CONVERTOR(aten::fmod, NONE, createMod, {PARAM(0) COMMA PARAM(1)} COMMA 1)

// full_like(Tensor self, Scalar fill_value, *, MemoryFormat? memory_format=None) -> Tensor
// full_like.dtype(Tensor self, Scalar fill_value, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=None) -> Tensor
OP_CONVERTOR(aten::full_like, NONE, createExpand, {PARAM(1) COMMA TENSOR_SHAPE_AS_IR(0)})

// full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
OP_CONVERTOR_WITH_CAST(aten::full, NONE, createExpand, {PARAM(1) COMMA HANDLE_LIST_AS_IR_CONSTANT(0)}, GET_RETURN_TYPE)

// Random Sampling ops
// aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor
OP_CONVERTOR(aten::rand, NONE, createRandomUniform, NULL COMMA OUTPUT_TENSOR_SHAPE(0) COMMA 1.0f COMMA 0.0f COMMA GET_RETURN_TYPE)
// aten::uniform_(Tensor self, float low, float high, Generator?) -> Tensor
OP_CONVERTOR(aten::uniform_, NONE, createRandomUniform, PARAM(0) COMMA TENSOR_SHAPE(0) COMMA CONSTANT_TO_FLOAT(2) COMMA CONSTANT_TO_FLOAT(1))
// aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor
OP_CONVERTOR(aten::randn, NONE, createRandomNormal, {} COMMA OUTPUT_TENSOR_SHAPE(0) COMMA 0.0f COMMA 1.0f COMMA GET_RETURN_TYPE)

OP_CONVERTOR_POP(symbols::poptorch::ipu_print_tensor, NONE, createPrintIpuTensor, PARAM(0) COMMA CONSTANT_TO_STRING(1))
OP_CONVERTOR_POP(symbols::poptorch::set_available_memory, NONE, createSetAvailableMemory, PARAM(0) COMMA CONSTANT_TO_FLOAT(1))
OP_CONVERTOR_POP(symbols::poptorch::begin_ipu_block, NONE, createBeginIpuBlock, CONSTANT_TO_LONG(0) COMMA CONSTANT_TO_LONG(1) COMMA CONSTANT_TO_LONG(2))
OP_CONVERTOR_POP(symbols::poptorch::set_matmul_serialization, NONE, createSetMatMulSerialization, PARAM(0) COMMA CONSTANT_TO_STRING(1) COMMA CONSTANT_TO_LONG(2) COMMA CONSTANT_TO_LONG(3))
OP_CONVERTOR_POP(symbols::poptorch::optimizer_group, NONE, createOptimizerGroup, CONSTANT_TO_LONG(0) COMMA HANDLE_TENSOR_LIST(1))
OP_CONVERTOR_POP(symbols::poptorch::recomputation_checkpoint, NONE, createRecomputationCheckpoint, PARAM(0))

OP_CONVERTOR(aten::pixel_shuffle, NONE, createDepthtospace, {PARAM(0)} COMMA CONSTANT_TO_LONG(1) COMMA "CRD")
