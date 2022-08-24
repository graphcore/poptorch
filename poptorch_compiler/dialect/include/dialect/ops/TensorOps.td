// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

def Poptorch_alias: Poptorch_Op<"alias", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input), [{
        $_state.addOperands({input});
        $_state.addTypes(input.getType());
     }]>
    ];
}

def Poptorch_detach: Poptorch_NotImplementedOp<"detach", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input), [{
        $_state.addOperands({input});
        $_state.addTypes(input.getType());
     }]>
    ];
}

def Poptorch_empty_tensor: Poptorch_Op<"empty_tensor", []> {
    let arguments = (ins I64ArrayAttr:$size, TypeAttr:$type);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "const std::vector<std::int64_t>&":$size, "mlir::Type":$type),[{
        $_state.addAttribute("size", $_builder.getI64ArrayAttr(size));
        $_state.addAttribute("type", mlir::TypeAttr::get(type));
        $_state.addTypes(mlir::RankedTensorType::get(size, type));
     }]>];
}

def Poptorch_fill_: Poptorch_Op<"fill_", []> {
    let arguments = (ins Poptorch_tensor:$input, F32Attr:$value);

    let builders = [OpBuilder<(ins "mlir::ValueRange":$values,
                                      "float":$value),[{
        $_state.addOperands(values);
        $_state.addAttribute("value", $_builder.getF32FloatAttr(value));
     }]>
    ];
}

def Poptorch_zero_: Poptorch_Op<"zero_", []> {
    let arguments = (ins Poptorch_tensor:$input);
}

def Poptorch_triu: Poptorch_NotImplementedOp<"triu", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$diagonal);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::int64_t":$diagonal),[{
    $_state.addOperands({self});
    $_state.addAttribute("diagonal", $_builder.getI64IntegerAttr(diagonal));
    $_state.addTypes(self.getType());
  }]>];
}

// to.dtype(Tensor(a) self, ScalarType dtype, ...) -> Tensor(a)
def Poptorch_cast : Poptorch_Op<"cast", []> {
  let arguments = (ins Poptorch_tensor:$self, TypeAttr:$dtype);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "mlir::Type":$dtype), [{
        $_state.addOperands({self});
        $_state.addAttribute("dtype", mlir::TypeAttr::get(dtype));

        // Keep shape but overwrite type
        $_state.addTypes(mlir::RankedTensorType::get(
            self.getType().cast<mlir::RankedTensorType>().getShape(), dtype));
  }]>];
}

def Poptorch_copy_: Poptorch_Op<"copy_", []> {
  let arguments = (ins Poptorch_tensor:$self, Poptorch_tensor:$src);
  let hasCanonicalizer = 1; // See CanonicalizeInplaceCopies.cpp
}


def Poptorch_print_tensor : Poptorch_Op<"print_tensor", []> {
  let arguments = (ins Poptorch_tensor:$input, StrAttr:$title);
}

def Poptorch_tensorconstant_float : Poptorch_Op<"tensorconstant_float", []> {
    let arguments = (ins F32ArrayAttr:$data,
                         I64ArrayAttr:$shape);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "const std::vector<float>&":$data,
                                   "const std::vector<std::int64_t>&":$shape),[{
        $_state.addAttribute("data", $_builder.getF32ArrayAttr(data));
        $_state.addAttribute("shape", $_builder.getI64ArrayAttr(shape));
        $_state.addTypes(mlir::RankedTensorType::get(shape, $_builder.getF32Type()));
     }]>];
}

def Poptorch_tensorconstant_int : Poptorch_Op<"tensorconstant_int", []> {
    let arguments = (ins I32ArrayAttr:$data,
                         I64ArrayAttr:$shape);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "const std::vector<std::int32_t>&":$data,
                                   "const std::vector<std::int64_t>&":$shape),[{
        $_state.addAttribute("data", $_builder.getI32ArrayAttr(data));
        $_state.addAttribute("shape", $_builder.getI64ArrayAttr(shape));
        $_state.addTypes(mlir::RankedTensorType::get(shape, $_builder.getIntegerType(32, true)));
     }]>];
}

def Poptorch_concat: Poptorch_Op<"concat", [ImplicitCastOperand<0>]> {

    let arguments = (ins Poptorch_tensorlist:$tensors,
                         I64Attr:$dim);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::ValueRange":$tensors,
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

// PyTorch operation:
/* func: topk.values(Tensor self, int k, int dim=-1,
                    bool largest=True, bool sorted=True,
                    Tensor(a!) values, Tensor(b!) indices) 
                    -> (Tensor(a!) , Tensor(b!))*/
def Poptorch_topk : Poptorch_Op<"topk", []> {
    let arguments = (ins Poptorch_tensor:$self, 
                         I64Attr:$K,
                         I64Attr:$dim,
                         BoolAttr:$largest,
                         BoolAttr:$sorted);


    let results = (outs Poptorch_tensor:$values, Poptorch_tensor_no_grad:$indices);

    let builders = [OpBuilder<(ins "mlir::Value":$self, 
                                      "std::int64_t":$K,
                                      "std::int64_t":$dim,
                                      "bool":$largest,
                                      "bool":$sorted),[{
        // Add op
        $_state.addOperands({self});

        mlir::RankedTensorType input_shape = self.getType().cast<mlir::RankedTensorType>();

        llvm::SmallVector<std::int64_t, 4> out_shape {input_shape.getShape().begin(),
                                                      input_shape.getShape().end()};

        if ( dim < 0) {
            dim = out_shape.size() + dim;
        }

        // Add attrs
        $_state.addAttribute("K", $_builder.getI64IntegerAttr(K));
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("largest", $_builder.getBoolAttr(largest));
        $_state.addAttribute("sorted", $_builder.getBoolAttr(sorted));

        // add returns
        out_shape[dim] = K;

        auto out_type = mlir::RankedTensorType::get(out_shape, input_shape.getElementType());
        auto out_indices = mlir::RankedTensorType::get(out_shape, $_builder.getIntegerType(32 /*width*/, true /*signed*/));

        $_state.addTypes({out_type, out_indices});
     }]>
    ];
}


class Poptorch_reduce_single_dim<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$self, OptionalAttr<I64Attr>:$dim, BoolAttr:$keepdim);

    let results = (outs Poptorch_tensor_no_grad:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "std::optional<std::int64_t>":$dim,
                                      "std::int64_t":$keepdim),[{
        $_state.addOperands(self);

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        if (dim.has_value()) {
            *dim = convertToPositiveDim(*dim, tensor.getShape().size());

            $_state.addAttribute("dim", $_builder.getI64IntegerAttr(*dim));
        }
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        auto getShape = [&]() {
            // We are flattenning the dimension we're passed
            if (dim.has_value()) {
                auto ref = tensor.getShape();

                llvm::SmallVector<std::int64_t, 4> shape{ref.begin(), ref.end()};

                if (!shape.empty()) {
                    if (!keepdim) {
                        shape.erase(shape.begin() + *dim);
                    } else {
                        shape[*dim] = 1;
                    }
                }

                return shape;
            } else {
                return llvm::SmallVector<std::int64_t, 4>{};
            }
        };

        $_state.addTypes(mlir::RankedTensorType::get(getShape(), $_builder.getIntegerType(32, true)));
     }]>
    ];
}

def Poptorch_argmax_out : Poptorch_reduce_single_dim<"argmax_out"> {}
def Poptorch_argmin_out : Poptorch_reduce_single_dim<"argmin_out"> {}

def Poptorch_argsort : Poptorch_NotImplementedOp<"argsort", []> {
    let arguments = (ins Poptorch_tensor:$self, I64Attr:$dim, BoolAttr:$descending);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "std::int64_t":$dim,
                                      "std::int64_t":$descending),[{
        $_state.addOperands(self);
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("descending", $_builder.getBoolAttr(descending));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        $_state.addTypes(mlir::RankedTensorType::get(tensor.getShape(), $_builder.getIntegerType(32, true)));
     }]>
    ];
}

def Poptorch_dropout : Poptorch_Op<"dropout", []> {
    let arguments = (ins Poptorch_tensor:$input, F32Attr:$p, BoolAttr:$training);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input, "float":$p, "bool":$training),[{
        $_state.addOperands({input});
        $_state.addAttribute("p", $_builder.getF32FloatAttr(p));
        $_state.addAttribute("training", $_builder.getBoolAttr(training));
        $_state.addTypes({input.getType()});
      }]>
    ];
}

class Poptorch_dropout_NotImplementedOps<string name, list<OpTrait> traits = []> : Poptorch_NotImplementedOp<name, traits> {
    let arguments = (ins Poptorch_tensor:$input, F32Attr:$p, BoolAttr:$training);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input, "float":$p, "bool":$training),[{
        $_state.addOperands({input});
        $_state.addAttribute("p", $_builder.getF32FloatAttr(p));
        $_state.addAttribute("training", $_builder.getBoolAttr(training));
        $_state.addTypes({input.getType()});
      }]>
    ];
}

def Poptorch_feature_dropout : Poptorch_dropout_NotImplementedOps<"feature_dropout", []> {}
def Poptorch_feature_dropout_ : Poptorch_dropout_NotImplementedOps<"feature_dropout_", []> {}

def Poptorch_where : Poptorch_Op<"where", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$condition,
                       Poptorch_tensor:$self,
                       Poptorch_tensor:$other);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$condition,
                                    "mlir::Value":$self,
                                    "mlir::Value":$other), [{
          $_state.addOperands({condition, self, other});
          $_state.addTypes({self.getType()});

  }]>];
}

def Poptorch_zeros_like : Poptorch_Op<"zeros_like", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self, OptionalAttr<TypeAttr>:$dtype);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self, "std::optional<mlir::Type>":$dtype), [{
    $_state.addOperands(self);
    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

    auto out_type = dtype.value_or(tensor.getElementType());
    $_state.addAttribute("dtype", mlir::TypeAttr::get(out_type));

    $_state.addTypes(mlir::RankedTensorType::get(tensor.getShape(), out_type));
  }]>];
}

def Poptorch_full_like : Poptorch_NotImplementedOp<"full_like", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self, F32Attr:$fill_value);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self, "float":$fill_value), [{
    $_state.addOperands(self);
    $_state.addAttribute("fill_value", $_builder.getF32FloatAttr(fill_value));

    $_state.addTypes({self.getType()});
  }]>];
}

def Poptorch_one_hot : Poptorch_NotImplementedOp<"one_hot", []> {
  let arguments = (ins Poptorch_tensor:$self, I64Attr:$num_classes);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self, "std::int64_t":$num_classes), [{
    $_state.addOperands(self);

    ERROR_ON_MSG(num_classes == -1,
                "OneHot num classes must be specified and must be constant.");

    $_state.addAttribute("num_classes", $_builder.getI64IntegerAttr(num_classes));

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

    auto ref = tensor.getShape();
    llvm::SmallVector<std::int64_t, 4> shape{ref.begin(), ref.end()};
    shape.push_back(num_classes);

    $_state.addTypes(mlir::RankedTensorType::get(shape, $_builder.getIntegerType(32, true)));
  }]>];
}

def Poptorch_arange : Poptorch_NotImplementedOp<"arange", []> {
  let arguments = (ins F32Attr:$start,
                       F32Attr:$end,
                       F32Attr:$step,
                       OptionalAttr<TypeAttr>:$dtype);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "float":$start,
                                 "float":$end,
                                 "float":$step,
                                 "std::optional<mlir::Type>":$dtype), [{
    $_state.addAttribute("start", $_builder.getF32FloatAttr(start));
    $_state.addAttribute("end", $_builder.getF32FloatAttr(end));
    $_state.addAttribute("step", $_builder.getF32FloatAttr(step));

    auto out_type = dtype.value_or($_builder.getF32Type());
    $_state.addAttribute("dtype", mlir::TypeAttr::get(out_type));

    const auto count = static_cast<int64_t>((end - start) / step);

    $_state.addTypes(mlir::RankedTensorType::get({count}, out_type));
  }]>];
}

// ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
def Poptorch_ones: Poptorch_NotImplementedOp<"ones", []> {
  let arguments = (ins I64ArrayAttr:$size,
                       OptionalAttr<TypeAttr>:$dtype);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "const std::vector<std::int64_t>&":$size,
                                 "std::optional<mlir::Type>":$dtype), [{
    $_state.addAttribute("size", $_builder.getI64ArrayAttr(size));
    auto out_type = dtype.value_or($_builder.getF32Type());
    $_state.addAttribute("dtype", mlir::TypeAttr::get(out_type));

    $_state.addTypes(mlir::RankedTensorType::get(size, out_type));
  }]>];
}

def Poptorch_masked_fill_Scalar : Poptorch_NotImplementedOp<"masked_fill_Scalar", []> {
  let arguments = (ins Poptorch_tensor:$self, Poptorch_tensor:$mask, F32Attr:$scalar);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self, "mlir::Value":$mask, "float":$scalar), [{
    $_state.addOperands({self, mask});
    $_state.addAttribute("scalar", $_builder.getF32FloatAttr(scalar));


    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    mlir::RankedTensorType mask_tensor = mask.getType().cast<mlir::RankedTensorType>();

    $_state.addTypes(mlir::RankedTensorType::get(broadcast(tensor.getShape(), mask_tensor.getShape()), tensor.getElementType()));
  }]>];
}

def Poptorch_flip : Poptorch_NotImplementedOp<"flip", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self, I64ArrayAttr:$dims);
  let results = (outs Poptorch_tensor:$out);
  let builders = [
    OpBuilder<(ins "mlir::Value":$self, "const std::vector<std::int64_t>&":$dims), [{
      $_state.addOperands(self);
      $_state.addAttribute("dims", $_builder.getI64ArrayAttr(dims));

      $_state.addTypes(self.getType());
    }]>
  ];
}

def Poptorch_roll : Poptorch_NotImplementedOp<"roll", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self, I64ArrayAttr:$shifts, I64ArrayAttr:$dims);
  let results = (outs Poptorch_tensor:$out);
  let builders = [
    OpBuilder<(ins "mlir::Value":$self,
               "const std::vector<std::int64_t>&":$shifts,
               "const std::vector<std::int64_t>&":$dims), [{
      $_state.addOperands(self);
      $_state.addAttribute("shifts", $_builder.getI64ArrayAttr(shifts));
      $_state.addAttribute("dims", $_builder.getI64ArrayAttr(dims));

      $_state.addTypes(self.getType());
    }]>
  ];
}

def Poptorch_resize_ : Poptorch_NotImplementedOp<"resize_", []> {
  let arguments = (ins Poptorch_tensor:$self, I64ArrayAttr:$size);
  let results = (outs Poptorch_tensor:$out);
  let builders = [
    OpBuilder<(ins "mlir::Value":$self,
                   "const std::vector<std::int64_t>&":$size), [{
      $_state.addOperands(self);
      $_state.addAttribute("size", $_builder.getI64ArrayAttr(size));
      $_state.addTypes(mlir::RankedTensorType::get(size, getElementType(self)));
    }]>
  ];
}
