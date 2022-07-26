// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
/*
 * Random number generation operations.
 */


def Poptorch_normal_ : Poptorch_Op<"normal_", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self,
                       F32Attr:$mean,
                       F32Attr:$stdv);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "float":$mean,
                                    "float":$stdv), [{
        $_state.addOperands({self});
        $_state.addAttribute("mean", $_builder.getF32FloatAttr(mean));
        $_state.addAttribute("stdv", $_builder.getF32FloatAttr(stdv));
        $_state.addTypes({self.getType()});
  }]>];
}

def Poptorch_normal_Tensor_Tensor : Poptorch_Op<"normal_Tensor_Tensor",
                                                [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$means,
                       Poptorch_tensor:$stdvs);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$means,
                                    "mlir::Value":$stdvs), [{
        $_state.addOperands({means, stdvs});
        $_state.addTypes({means.getType()});
  }]>];
}

def Poptorch_normal_Tensor_float : Poptorch_Op<"normal_Tensor_float",
                                               [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$means,
                       F32Attr:$stdv);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$means,
                                    "float":$stdv), [{
        $_state.addOperands({means});
        $_state.addAttribute("stdv", $_builder.getF32FloatAttr(stdv));
        $_state.addTypes({means.getType()});
  }]>];
}

def Poptorch_normal_float_Tensor : Poptorch_Op<"normal_float_Tensor",
                                               [SameOperandsAndResultShape]> {
  let arguments = (ins F32Attr:$mean,
                       Poptorch_tensor:$stdvs);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "float":$mean,
                                    "mlir::Value":$stdvs), [{
        $_state.addOperands({stdvs});
        $_state.addAttribute("mean", $_builder.getF32FloatAttr(mean));
        $_state.addTypes({stdvs.getType()});
  }]>];
}

def Poptorch_uniform_ : Poptorch_Op<"uniform_", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self,
                       F32Attr:$from,
                       F32Attr:$to);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "float":$from,
                                    "float":$to), [{
        $_state.addOperands({self});
        $_state.addAttribute("from", $_builder.getF32FloatAttr(from));
        $_state.addAttribute("to", $_builder.getF32FloatAttr(to));
        $_state.addTypes({self.getType()});
  }]>];
}

def Poptorch_random_ : Poptorch_Op<"random_", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self), [{
        $_state.addOperands({self});
        $_state.addTypes({self.getType()});
  }]>];
}

def Poptorch_random__from : Poptorch_Op<"random__from", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$from,
                       OptionalAttr<I64Attr>:$to);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "int64_t":$from,
                                    "std::optional<int64_t>":$to), [{
        $_state.addOperands({self});

        $_state.addAttribute("from", $_builder.getI64IntegerAttr(from));
        if (to)
          $_state.addAttribute("to", $_builder.getI64IntegerAttr(*to));

        $_state.addTypes({self.getType()});
  }]>];
}

def Poptorch_exponential_ : Poptorch_Op<"exponential_", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self,
                       F32Attr:$lambd);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "float":$lamb), [{
        $_state.addOperands({self});
        $_state.addAttribute("lambd", $_builder.getF32FloatAttr(lamb));
        $_state.addTypes({self.getType()});
  }]>];
}

def Poptorch_bernoulli : Poptorch_NotImplementedOp<"bernoulli", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$input);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$input), [{
        $_state.addOperands({input});

        $_state.addTypes({input.getType()});
  }]>];
}

def Poptorch_bernoulli__float : Poptorch_Op<"bernoulli__float", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$input,
                       F32Attr:$prob);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$input,
                                    "float":$prob), [{
        $_state.addOperands({input});
        $_state.addAttribute("prob", $_builder.getF32FloatAttr(prob));

        mlir::RankedTensorType input_tensor = input.getType().cast<mlir::RankedTensorType>();
        llvm::SmallVector<std::int64_t, 4> input_shape {input_tensor.getShape().begin(), input_tensor.getShape().end()};

        $_state.addTypes(mlir::RankedTensorType::get(input_shape, $_builder.getF32Type()));
  }]>];
}

def Poptorch_bernoulli__tensor : Poptorch_Op<"bernoulli__tensor", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$input,
                       Poptorch_tensor:$probs);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$input,
                                    "mlir::Value":$probs), [{
        $_state.addOperands({input, probs});

        mlir::RankedTensorType input_tensor = input.getType().cast<mlir::RankedTensorType>();
        llvm::SmallVector<std::int64_t, 4> input_shape {input_tensor.getShape().begin(), input_tensor.getShape().end()};

        $_state.addTypes(mlir::RankedTensorType::get(input_shape, $_builder.getF32Type()));
  }]>];
}

def Poptorch_bernoulli_out : Poptorch_Op<"bernoulli_out", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self), [{
        $_state.addOperands({self});

        mlir::RankedTensorType self_tensor = self.getType().cast<mlir::RankedTensorType>();
        llvm::SmallVector<std::int64_t, 4> self_shape {self_tensor.getShape().begin(), self_tensor.getShape().end()};

        $_state.addTypes(mlir::RankedTensorType::get(self_shape, $_builder.getF32Type()));
  }]>];
}

def Poptorch_randperm : Poptorch_Op<"randperm", []> {
  let arguments = (ins I64Attr:$n);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "int64_t":$n), [{
        $_state.addAttribute("n", $_builder.getI64IntegerAttr(n));

        $_state.addTypes(mlir::RankedTensorType::get({n}, $_builder.getIntegerType(32, true)));
  }]>];
}
