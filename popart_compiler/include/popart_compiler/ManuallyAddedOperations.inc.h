OP_DECL("popart::reshape_static_shape", reshape, impl->reshape, ARG(INT_VEC, shape),
        BODY_ARG(shape))
OP_DECL("poptorch::ipu_print_tensor", ipu_print_tensor,
        AiGraphcoreOpset1.printtensor, NONE, NONE)
OP_DECL("poptorch::int_constant", intConstant, impl->intConstant,
        ARG(INT_VEC, data) ARG(INT_VEC, shape), BODY_ARG(data) BODY_ARG(shape))
OP_DECL("poptorch::float_constant", floatConstant, impl->floatConstant,
        ARG(FLOAT_VEC, data) ARG(INT_VEC, shape),
        BODY_ARG(data) BODY_ARG(shape))
