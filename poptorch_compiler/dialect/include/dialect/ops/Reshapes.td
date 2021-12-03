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


class Poptorch_reshape_generic<string name> : Poptorch_Op<name, [ViewOp]> {
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

// Reshape and expand have the exact same setup but reshape is just a reshape while exapand is a broadcast under the hood.
def Poptorch_reshape : Poptorch_reshape_generic<"reshape"> {}
def Poptorch_expand : Poptorch_reshape_generic<"expand"> {}
