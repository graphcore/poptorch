// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> (Tensor(a))
def Poptorch_as_strided: Poptorch_Op<"as_strided", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$size, I64ArrayAttr:$strides);

    let results = (outs Poptorch_tensor:$result);


    let builders = [OpBuilderDAG<(ins "mlir::Value":$val1,
                                      "const std::vector<std::int64_t>&":$size,
                                      "const std::vector<std::int64_t>&":$strides),[{
        $_state.addOperands({val1});
        $_state.addAttribute("size", $_builder.getI64ArrayAttr(size));
        $_state.addAttribute("strides", $_builder.getI64ArrayAttr(strides));

        mlir::RankedTensorType tensor = val1.getType().cast<mlir::RankedTensorType>();
        $_state.addTypes(mlir::RankedTensorType::get(size, tensor.getElementType()));
     }]>];
}



// aten::transpose.int(Tensor(a) self, int dim0, int dim1)
def Poptorch_transpose: Poptorch_Op<"transpose", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim0,  I64Attr:$dim1);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "std::int64_t":$dim0, "std::int64_t":$dim1),[{
        $_state.addOperands({value});
        $_state.addAttribute("dim0", $_builder.getI64IntegerAttr(dim0));
        $_state.addAttribute("dim1", $_builder.getI64IntegerAttr(dim1));

        // Transpose the type.
        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        llvm::SmallVector<std::int64_t, 4> shape = {t1.getShape().begin(),  t1.getShape().end()};

        if (dim0 < shape.size() && dim1 < shape.size()) {
            std::swap(shape[dim0], shape[dim1]);
        }

        $_state.addTypes(mlir::RankedTensorType::get(shape, t1.getElementType()));
     }]>
    ];
}


def Poptorch_reshape: Poptorch_Op<"reshape", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$shape);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "const std::vector<std::int64_t>&":$shape),[{
        $_state.addOperands({value});

        // At most one element can be -1, for which the shape will need to be
        // resolved based on the number of elements.
        const int64_t no_dim = -1;
        int64_t inferred_dim = no_dim;

        // Store the product of the shape (excluding the -1 dimension)
        uint64_t shape_product = 1;

        for(size_t dim_idx = 0; dim_idx < shape.size(); dim_idx++) {
            if(shape[dim_idx] == -1) {
                if(inferred_dim != no_dim) {
                    // Match PyTorch's error message
                    ERROR("only one dimension can be inferred");
                }
                inferred_dim = dim_idx;
            } else if(shape[dim_idx] <= 0) {
                // This should only occur as an internal error.
                ERROR("invalid shape dimension, " << shape[dim_idx]
                      << ", in Poptorch_reshape_generic");
            } else {
                shape_product *= shape[dim_idx];
            }
        }

        // Obtain the tensor type and clone the output shape in case we modify
        // it when resolving the "-1".
        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        auto numel_in = t1.getNumElements();

        // Clone the const shape in case we need to resolve a -1 dim
        auto shape_out = shape;

        // Resolve the -1 dim if required
        if (inferred_dim != no_dim) {
            if(numel_in % shape_product == 0) {
                shape_out[inferred_dim] = numel_in/shape_product;
                shape_product = numel_in;
            }
        }

        // At this stage, shape_product should be numel_in, or the shape is
        // invalid (with or without a -1 dim)
        if (shape_product != numel_in) {
            std::ostringstream err;
            err << "shape'[";

            for(size_t dim_idx = 0; dim_idx < shape.size(); dim_idx++) {
                err << shape[dim_idx];
                if(dim_idx + 1 != shape.size()) {
                    err << ", ";
                }
            }

            err << "]' is invalid for input of size ";
            err << numel_in;
            throw std::runtime_error(err.str());
        }

        // Otherwise, the reshape is succesful
        // Ensure the attribute is the resolved shape (without -1)
        $_state.addAttribute("shape", $_builder.getI64ArrayAttr(shape_out));

        mlir::Type e1 = t1.getElementType();
        $_state.addTypes(mlir::RankedTensorType::get(shape_out, e1));
     }]>
    ];
}


// Expand is like reshape but can broadcast unity dimensions and sets -1
// dimensions to the same as the input. New dimensions are added at the front
// TODO T52507 Fully inplement expand
def Poptorch_expand: Poptorch_Op<"expand", [ViewOp]> {
      let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$shape);
  let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "const std::vector<std::int64_t>&":$shape),[{
        $_state.addOperands({value});
        $_state.addAttribute("shape", $_builder.getI64ArrayAttr(shape));

        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        mlir::Type e1 = t1.getElementType();
        $_state.addTypes(mlir::RankedTensorType::get(shape, e1));
     }]>
    ];
}


// squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
def Poptorch_squeeze_dim: Poptorch_Op<"squeeze_dim", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "std::int64_t":$dim),[{
        $_state.addOperands({value});
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        mlir::Type e1 = t1.getElementType();
        auto shape = t1.getShape();
        if (dim < 0) {
            dim += shape.size();
        }
        std::vector<int64_t> new_shape = shape;
        if (shape[dim] == 1) {
            new_shape.erase(new_shape.begin() + dim);
        }
        $_state.addTypes(mlir::RankedTensorType::get(new_shape, e1));
     }]>
    ];
}