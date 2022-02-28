// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
/*
 * Random number generation operations.
 */


def Poptorch_normal_ : Poptorch_Op<"normal_", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$self,
                       F32Attr:$mean,
                       F32Attr:$stdv);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
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

  let builders = [OpBuilderDAG<(ins "mlir::Value":$means,
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

  let builders = [OpBuilderDAG<(ins "mlir::Value":$means,
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

  let builders = [OpBuilderDAG<(ins "float":$mean,
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

  let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                    "float":$from,
                                    "float":$to), [{
        $_state.addOperands({self});
        $_state.addAttribute("from", $_builder.getF32FloatAttr(from));
        $_state.addAttribute("to", $_builder.getF32FloatAttr(to));
        $_state.addTypes({self.getType()});
  }]>];
}
