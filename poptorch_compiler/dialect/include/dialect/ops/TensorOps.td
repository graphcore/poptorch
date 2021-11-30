
def Poptorch_empty_tensor: Poptorch_Op<"empty_tensor", []> {
    let arguments = (ins I64ArrayAttr:$size, TypeAttr:$type);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "const std::vector<std::int64_t>&":$size, "mlir::Type":$type),[{
        $_state.addAttribute("size", $_builder.getI64ArrayAttr(size));
        $_state.addAttribute("type", mlir::TypeAttr::get(type));
        $_state.addTypes(mlir::RankedTensorType::get(size, type));
     }]>];
}

def Poptorch_fill_: Poptorch_Op<"fill_", []> {
    let arguments = (ins Poptorch_tensor:$input, F32Attr:$value);

    let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values,
                                      "float":$value),[{
        $_state.addOperands(values);
        $_state.addAttribute("value", $_builder.getF32FloatAttr(value));
     }]>
    ];
}

def Poptorch_zero_: Poptorch_Op<"zero_", []> {
    let arguments = (ins Poptorch_tensor:$input);
}

def Poptorch_copy_: Poptorch_Op<"copy_", []> {
  let arguments = (ins Poptorch_tensor:$self, Poptorch_tensor:$src);
}


def Poptorch_print_tensor : Poptorch_Op<"print_tensor", []> {
  let arguments = (ins Poptorch_tensor:$input, StrAttr:$title);
}


def Poptorch_tensorconstant_float : Poptorch_Op<"tensorconstant_float", []> {
    let arguments = (ins F32ArrayAttr:$data);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "const std::vector<float>&":$data),[{
        $_state.addAttribute("data", $_builder.getF32ArrayAttr(data));
        $_state.addTypes(mlir::RankedTensorType::get(1, $_builder.getF32Type()));
     }]>];
}

def Poptorch_tensorconstant_int : Poptorch_Op<"tensorconstant_int", []> {
    let arguments = (ins I32ArrayAttr:$data);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "const std::vector<std::int32_t>&":$data),[{
        $_state.addAttribute("data", $_builder.getI32ArrayAttr(data));
        $_state.addTypes(mlir::RankedTensorType::get(1, $_builder.getIntegerType(32, true)));
     }]>];
}

def Poptorch_concat: Poptorch_Op<"concat", []> {

    let arguments = (ins Poptorch_tensorlist:$tensors,
                         I64Attr:$dim);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$tensors,
                                      "std::uint64_t":$dim),[{
        $_state.addOperands(tensors);

        mlir::RankedTensorType first_type = tensors[0].getType().cast<mlir::RankedTensorType>();

        llvm::SmallVector<std::int64_t, 4> out_shape {first_type.getShape().begin(),
                                                      first_type.getShape().end()};

        if (dim < 0) {
          dim += out_shape.size();
        }

        for (auto i = 1u; i < tensors.size(); i++) {
          out_shape[dim] += tensors[i].getType().cast<mlir::RankedTensorType>().getDimSize(dim);
        }

        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

        auto out_type = mlir::RankedTensorType::get(out_shape, first_type.getElementType());

        $_state.addTypes({out_type});
     }]>
    ];
}

def Poptorch_dropout : Poptorch_Op<"dropout", []> {
    let arguments = (ins Poptorch_tensor:$input, F32Attr:$p, BoolAttr:$training);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$input, "float":$p, "bool":$training),[{
        $_state.addOperands({input});
        $_state.addAttribute("p", $_builder.getF32FloatAttr(p));
        $_state.addAttribute("training", $_builder.getBoolAttr(training));
        $_state.addTypes({input.getType()});
      }]>
    ];
}