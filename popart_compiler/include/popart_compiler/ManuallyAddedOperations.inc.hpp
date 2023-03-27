// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
OP_DECL(popart, reshape_static_shape, reshape, _impl->reshape,
        ARG(INT_VEC, shape), BODY_ARG(shape))
OP_DECL(poptorch, ipu_print_tensor, ipu_print_tensor, AiGraphcoreOpset1.printtensor,
        ARG(INT,print_gradient) ARG(STRING,title) ARG(INT,summariseThreshold) ARG(INT,edgeItems)
        ARG(INT,maxLineWidth) ARG(INT,digits) ARG(INT,floatFormat) ARG(CHAR,separator) ARG(CHAR,openBracket) ARG(CHAR,closeBracket) ,
        BODY_ARG(print_gradient) BODY_ARG(DEBUG_CONTEXT("Printtensor"))BODY_ARG(title) BODY_ARG(summariseThreshold) BODY_ARG(edgeItems)
        BODY_ARG(maxLineWidth) BODY_ARG(digits) BODY_ARG(floatFormat) BODY_ARG(separator) BODY_ARG(openBracket) BODY_ARG(closeBracket))
OP_DECL(poptorch, tensor_constant, tensor_constant, _impl->tensorConstant,
        POPART_CONST_ARG(popartConstant), BODY_ARG(popartConstant))
OP_DECL(poptorch, host_side_tensor_constant, host_side_tensor_constant,
        _impl->hostSideTensorConstant,
        HOST_SIDE_CONST_ARG(hostSideTensorConstant),
        BODY_ARG(hostSideTensorConstant))

OP_DECL(poptorch, constant_pad, constant_pad, AiOnnxOpset11.pad,
        NONE, BODY_ARG("constant") BODY_ARG(DEBUG_CONTEXT("Constantpad")))
OP_DECL(poptorch, reflection_pad, reflection_pad, AiOnnxOpset11.pad,
        NONE, BODY_ARG("reflect"))
OP_DECL(poptorch, edge_pad, edge_pad, AiOnnxOpset11.pad, NONE, BODY_ARG("edge")
        BODY_ARG(DEBUG_CONTEXT("Reflectionpad")))

OP_DECL(poptorch, add_not_in_place, add_not_in_place, _impl->addNotInPlace,
        NONE, NONE)

OP_DECL(poptorch, custom_operation, custom_operation, _impl->customOperation,
        ARG(STRING, name) ARG(STRING, domain) ARG(INT, version)
            ARG(INT, num_outputs) POPART_ATTRIB_VEC_ARG(attributes),
        BODY_ARG(name) BODY_ARG(domain) BODY_ARG(version) BODY_ARG(num_outputs)
            BODY_ARG(attributes))

OP_DECL_NO_RETURN(poptorch, addOutputTensor, addOutputTensor,
                  _impl->addOutputTensor, NONE, NONE)

OP_DECL(poptorch, random_uniform, random_uniform, _impl->randomUniform,
        ARG(INT_VEC, shape) ARG(FLOAT, high) ARG(FLOAT, low) ARG(STRING, dtype),
        BODY_ARG(shape) BODY_ARG(high) BODY_ARG(low) BODY_ARG(dtype))

OP_DECL(poptorch, random_normal, random_normal, _impl->randomNormal,
        ARG(INT_VEC, shape) ARG(FLOAT, mean) ARG(FLOAT, scale)
            ARG(STRING, dtype),
        BODY_ARG(shape) BODY_ARG(mean) BODY_ARG(scale) BODY_ARG(dtype))

OP_DECL(poptorch, ones, ones, _impl->ones,
        ARG(INT_VEC, shape) ARG(STRING, dtype), BODY_ARG(shape) BODY_ARG(dtype))
OP_DECL(poptorch, zeros, zeros, _impl->zeros,
        ARG(INT_VEC, shape) ARG(STRING, dtype), BODY_ARG(shape) BODY_ARG(dtype))

OP_DECL(poptorch, recomputation_checkpoint, recomputation_checkpoint,
        _impl->recomputationCheckpoint, NONE, NONE)

OP_DECL(poptorch, unfold, unfold, _impl->unfold,
        ARG(INT, dimension) ARG(INT, size) ARG(INT, step),
        BODY_ARG(dimension) BODY_ARG(size) BODY_ARG(step))

OP_DECL(poptorch, prelu, prelu, _impl->prelu, NONE, NONE)

// Operations which need extra types

#define EMPTY_FLOAT_VEC std::vector<float>()
#define EMPTY_STRING_VEC std::vector<std::string>()
#define OPTIONAL_FLOAT nonstd::optional<float>()
#define OPTIONAL_INT nonstd::optional<int64_t>()

OP_DECL(poptorch, gru, gru, AiOnnxOpset11.gru, ARG(INT, hidden_size),
        BODY_ARG(2) BODY_ARG(EMPTY_FLOAT_VEC) BODY_ARG(EMPTY_FLOAT_VEC)
        BODY_ARG(EMPTY_STRING_VEC) BODY_ARG(OPTIONAL_FLOAT)
        BODY_ARG("forward") BODY_ARG(hidden_size) BODY_ARG(1)
        BODY_ARG(DEBUG_CONTEXT("Gru")))

OP_DECL(poptorch, rnn, rnn, AiOnnxOpset11.rnn, ARG(STRING_VEC, activations),
        BODY_ARG(2) BODY_ARG(EMPTY_FLOAT_VEC) BODY_ARG(EMPTY_FLOAT_VEC)
        BODY_ARG(activations) BODY_ARG(OPTIONAL_FLOAT) BODY_ARG("forward")
        BODY_ARG(OPTIONAL_INT)
        BODY_ARG(DEBUG_CONTEXT("Rnn")))


#undef EMPTY_STRING_VEC
#undef OPTIONAL_INT
#undef OPTIONAL_FLOAT
#undef EMPTY_FLOAT_VEC
