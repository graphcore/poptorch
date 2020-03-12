OP_DECL("popart::reshape", reshape, impl->reshape, ARG(INT_VEC, shape),
        BODY_ARG(shape), NONE)
OP_DECL("poptorch::ipu_print_tensor", printtensor, aiGraphcore.printtensor, NONE, NONE, NONE)
