

def Poptorch_empty_tensor: Poptorch_Op<"empty_tensor", []> {
    let arguments = (ins I64ArrayAttr:$size);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "const std::vector<std::int64_t>&":$size),[{
        $_state.addAttribute("size", $_builder.getI64ArrayAttr(size));
        $_state.addTypes(mlir::RankedTensorType::get(size, $_builder.getF32Type()));
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


def Poptorch_tensorconstant : Poptorch_Op<"tensorconstant", []> {
    let arguments = (ins F32ArrayAttr:$data);
    let results = (outs Poptorch_tensor:$result);


    let builders = [OpBuilderDAG<(ins "const std::vector<float>&":$data),[{
        $_state.addAttribute("data", $_builder.getF32ArrayAttr(data));
        $_state.addTypes(mlir::RankedTensorType::get(1, $_builder.getF32Type()));
     }]>
    ];

}
