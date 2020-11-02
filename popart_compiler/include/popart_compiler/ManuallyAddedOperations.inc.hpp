// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
OP_DECL(popart, reshape_static_shape, reshape, _impl->reshape, ARG(INT_VEC, shape),
        BODY_ARG(shape))
OP_DECL(poptorch, ipu_print_tensor, ipu_print_tensor,
        AiGraphcoreOpset1.printtensor, NONE, NONE)
OP_DECL(poptorch, tensor_constant, tensor_constant, _impl->tensorConstant,
        POPART_CONSTANT_ARG(popartConstant), BODY_ARG(popartConstant))
OP_DECL(poptorch, constant_pad, constant_pad, AiOnnxOpset10.pad,
        ARG(INT_VEC, pads) ARG(FLOAT, value),
        BODY_ARG(pads) BODY_ARG("constant") BODY_ARG(value))
OP_DECL(poptorch, reflection_pad, reflection_pad, AiOnnxOpset10.pad,
        ARG(INT_VEC, pads),
        BODY_ARG(pads) BODY_ARG("reflect"))
OP_DECL(poptorch, edge_pad, edge_pad, AiOnnxOpset10.pad,
        ARG(INT_VEC, pads),
        BODY_ARG(pads) BODY_ARG("edge"))

OP_DECL(poptorch, add_not_in_place, add_not_in_place, _impl->addNotInPlace, NONE, NONE)

OP_DECL(poptorch, custom_operation, custom_operation, _impl->customOperation,
        ARG(STRING, name) ARG(STRING, domain) ARG(INT, version) ARG(INT, num_outputs),
        BODY_ARG(name) BODY_ARG(domain) BODY_ARG(version) BODY_ARG(num_outputs))

OP_DECL(poptorch, random_uniform, random_uniform, _impl->randomUniform,
        ARG(INT_VEC, shape) ARG(FLOAT, high) ARG(FLOAT, low) ARG(STRING, dtype),
        BODY_ARG(shape) BODY_ARG(high) BODY_ARG(low) BODY_ARG(dtype))

OP_DECL(poptorch, random_normal, random_normal, _impl->randomNormal,
        ARG(INT_VEC, shape) ARG(FLOAT, mean) ARG(FLOAT, scale) ARG(STRING, dtype),
        BODY_ARG(shape) BODY_ARG(mean) BODY_ARG(scale) BODY_ARG(dtype))

OP_DECL(poptorch, ones, ones, _impl->ones,
        ARG(INT_VEC, shape) ARG(STRING, dtype),
        BODY_ARG(shape) BODY_ARG(dtype))
OP_DECL(poptorch, zeros, zeros, _impl->zeros,
        ARG(INT_VEC, shape) ARG(STRING, dtype),
        BODY_ARG(shape) BODY_ARG(dtype))