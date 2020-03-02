/*
    OP_DECLS are in the following form:
    OP_DECL(string, function, onnx implementation, arguments, body argument)
     - string is the string representation of the function
     - function is the actual op part of the <namespace>:<op> pair and will be
   used to name/call the given function.
     - Onnx implementation is the underlaying onnx function which will be
   called.
     - Arguments are the arguments to the op which will be parsed by different
   macros depending on which file this is in.
     - Body arguments are just the names of the arguments so they can be used in
   the cpp file.
     - Indexing (TODO REMOVE) this is to allow us to index into the output. We will have to change this in the future.
*/
OP_DECL("aten::t", t, aiOnnx.transpose, NONE, NONE, NONE)
OP_DECL("popart::matmul", matmul, aiOnnx.matmul, NONE, NONE, NONE)
OP_DECL("aten::relu", relu, aiOnnx.relu, NONE, NONE, NONE)
OP_DECL("aten::relu_", relu_, aiOnnx.relu, NONE, NONE, NONE)
OP_DECL("popart::add", add, aiOnnx.add, NONE, NONE, NONE)
OP_DECL("popart::flatten", flatten, aiOnnx.flatten, NONE, NONE, NONE)
OP_DECL("popart::convolution", convolution, aiOnnx.conv,
        ARG(INT_VEC, dilation) ARG(INT, group) ARG(INT_VEC, kernel_shape)
            ARG(INT_VEC, pads) ARG(INT_VEC, strides),
        BODY_ARG(dilation) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(pads)
            BODY_ARG(strides),
        NONE)
OP_DECL("popart::batchnorm", batchnorm, aiOnnx.batchnormalization,
        ARG(INT, num_outputs) ARG(FLOAT, epsilon) ARG(FLOAT, momentum),
        BODY_ARG(num_outputs) BODY_ARG(epsilon) BODY_ARG(momentum), [0])
OP_DECL("popart::gemm", gemm, aiOnnx.gemm,
        ARG(FLOAT, alpha) ARG(FLOAT, beta) ARG(INT, transA) ARG(INT, transB),
        BODY_ARG(alpha) BODY_ARG(beta) BODY_ARG(transA) BODY_ARG(transB), NONE)
OP_DECL("popart::average_pool", average_pool, aiOnnx.averagepool,
        ARG(INT_VEC, kernel_shape) ARG(INT, count_include_pad)
            ARG(INT_VEC, pads) ARG(INT_VEC, strides),
        BODY_ARG(kernel_shape) BODY_ARG(count_include_pad) BODY_ARG(pads)
            BODY_ARG(strides),
        NONE)
OP_DECL("popart::max_pool", max_pool, aiOnnx.maxpool,
        ARG(INT, num_outputs) ARG(INT_VEC, kernel_size) ARG(INT_VEC, padding)
            ARG(INT, storage_order) ARG(INT_VEC, strides),
        BODY_ARG(num_outputs) BODY_ARG(kernel_size) BODY_ARG(padding)
            BODY_ARG(storage_order) BODY_ARG(strides),
        [0])
OP_DECL("popart::softmax", softmax, aiOnnx.softmax, ARG(INT, dim),
        BODY_ARG(dim), NONE)
OP_DECL("popart::reshape", softmax, impl->reshape, ARG(INT_VEC, shape),
        BODY_ARG(shape), NONE)
OP_DECL("poptorch::ipu_print_tensor", printtensor, aiGraphcore.printtensor, NONE, NONE, NONE)
OP_DECL("popart::dropout", dropout, aiOnnx.dropout, ARG(INT, num_outputs) ARG(FLOAT, rate), BODY_ARG(num_outputs) BODY_ARG(rate), [0])