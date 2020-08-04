// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
OP_DECL(popart, reshape_static_shape, reshape, _impl->reshape, ARG(INT_VEC, shape),
        BODY_ARG(shape))
OP_DECL(poptorch, ipu_print_tensor, ipu_print_tensor,
        AiGraphcoreOpset1.printtensor, NONE, NONE)
OP_DECL(poptorch, int_constant, int_constant, _impl->intConstant,
        ARG(INT_VEC, data) ARG(INT_VEC, shape), BODY_ARG(int64ToInt32(data)) BODY_ARG(shape))
OP_DECL(poptorch, int64_constant, int64_constant, _impl->int64Constant,
        ARG(INT_VEC, data) ARG(INT_VEC, shape), BODY_ARG(data) BODY_ARG(shape))
OP_DECL(poptorch, float_constant, float_constant, _impl->floatConstant,
        ARG(FLOAT_VEC, data) ARG(INT_VEC, shape) ARG(INT, isHalf),
        BODY_ARG(data) BODY_ARG(shape) BODY_ARG(static_cast<bool>(isHalf)))
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

OP_DECL(poptorch, random_uniform, random_uniform, _impl->randomUniform, ARG(INT_VEC, shape),
        BODY_ARG(shape))
