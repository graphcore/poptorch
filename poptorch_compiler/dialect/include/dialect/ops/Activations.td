// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
include "ops/ElementWise.td"


/*
 * These activation functions share the same boilerplate as the unary element wise ops.
 */

// Outplace
def Poptorch_hardsigmoid: Poptorch_elem_unary<"hardsigmoid"> {}
def Poptorch_hardswish: Poptorch_elem_unary<"hardswish"> {}
def Poptorch_swish: Poptorch_elem_unary<"swish"> {}
def Poptorch_relu: Poptorch_elem_unary<"relu"> {}
def Poptorch_gelu: Poptorch_elem_unary<"gelu"> {}

def Poptorch_elu : Poptorch_Op<"elu", [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$in1, F32Attr:$alpha, F32Attr:$scale, F32Attr:$input_scale);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self, "float":$alpha, "float":$scale, "float":$input_scale),[{
        $_state.addOperands({self});
        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));
        $_state.addAttribute("scale",$_builder.getF32FloatAttr(scale));
        $_state.addAttribute("input_scale",$_builder.getF32FloatAttr(input_scale));

        $_state.addTypes(self.getType());
     }]>
    ];
}

def Poptorch_hardshrink : Poptorch_Op<"hardshrink", [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$in1, F32Attr:$lambd);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self, "float":$lambd),[{
        $_state.addOperands({self});
        $_state.addAttribute("lambd",$_builder.getF32FloatAttr(lambd));

        $_state.addTypes(self.getType());
     }]>
    ];
}

def Poptorch_softshrink_out : Poptorch_Op<"softshrink_out", [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$in1, F32Attr:$lambd);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self, "float":$lambd),[{
        $_state.addOperands({self});
        $_state.addAttribute("lambd",$_builder.getF32FloatAttr(lambd));

        $_state.addTypes(self.getType());
     }]>
    ];
}

def Poptorch_rrelu_with_noise : Poptorch_Op<"rrelu_with_noise", [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$self, Poptorch_tensor:$noise, F32Attr:$lower, F32Attr:$upper, BoolAttr:$training);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self, "mlir::Value":$noise, "float":$lower, "float":$upper, "bool":$training),[{
        $_state.addOperands({self, noise});
        $_state.addAttribute("lower",$_builder.getF32FloatAttr(lower));
        $_state.addAttribute("upper",$_builder.getF32FloatAttr(upper));
        $_state.addAttribute("training",$_builder.getBoolAttr(training));

        $_state.addTypes(self.getType());
     }]>
    ];
}

def Poptorch_leaky_relu : Poptorch_Op<"leaky_relu", [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$in1, F32Attr:$negative_slope);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self, "float":$negative_slope),[{
        $_state.addOperands({self});
        $_state.addAttribute("negative_slope",$_builder.getF32FloatAttr(negative_slope));

        $_state.addTypes(self.getType());
     }]>
    ];
}

def Poptorch_leaky_relu_backward : Poptorch_Op<"leaky_relu_backward", [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$grad_output,
                         Poptorch_tensor:$self,
                         F32Attr:$negative_slope,
                         BoolAttr:$self_is_result);
    let results = (outs Poptorch_tensor:$grad_input);

    let builders = [OpBuilder<(ins "mlir::Value":$grad_output,
                                   "mlir::Value":$self,
                                   "float":$negative_slope,
                                   "bool":$self_is_result),[{
        $_state.addOperands({grad_output, self});
        $_state.addAttribute("negative_slope", $_builder.getF32FloatAttr(negative_slope));
        $_state.addAttribute("self_is_result", $_builder.getBoolAttr(self_is_result));

        $_state.addTypes(self.getType());
     }]>
    ];
}

def Poptorch_prelu : Poptorch_Op<"prelu", []> {
    let arguments = (ins Poptorch_tensor:$self, Poptorch_tensor:$weight);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self, "mlir::Value":$weight),[{
        $_state.addOperands({self, weight});

        $_state.addTypes(self.getType());
     }]>
    ];
}

def Poptorch_log_sigmoid_forward : Poptorch_Op<"log_sigmoid_forward", []> {
    let arguments = (ins Poptorch_tensor:$in1);
    let results = (outs Poptorch_tensor:$output,
                        Poptorch_tensor:$buffer);

    let builders = [OpBuilder<(ins "mlir::Value":$self),[{
        $_state.addOperands({self});

        $_state.addTypes({self.getType(), self.getType()});
     }]>
    ];
}

// Softmax takes an additional "dim" argument.
def Poptorch_softmax : Poptorch_Op<"softmax", [SameOperandsAndResultShape]> {

  // For now we just ignore the float to half bit.
  let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim, BoolAttr:$half_to_float);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$input, "std::int64_t":$dim, "bool":$half_to_float), [{
        $_state.addOperands({input});

        dim = convertToPositiveDim(dim, getShape(input).size());
        $_state.addAttribute("dim",$_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("half_to_float",$_builder.getBoolAttr(half_to_float));
        $_state.addTypes(input.getType());
  }]>];
}


def Poptorch_logsoftmax : Poptorch_Op<"logsoftmax", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim, BoolAttr:$half_to_float);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$input, "std::int64_t":$dim, "bool":$half_to_float), [{
        $_state.addOperands({input});
        $_state.addAttribute("dim",$_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("half_to_float",$_builder.getBoolAttr(half_to_float));
        $_state.addTypes(input.getType());
  }]>];
}


def Poptorch_logsoftmax_backward : Poptorch_Op<"logsoftmax_backward", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$grad_output, Poptorch_tensor:$output, I64Attr:$dim, Poptorch_tensor:$self);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$grad_output, "mlir::Value":$output, "std::int64_t":$dim, "mlir::Value":$self), [{
        $_state.addOperands({grad_output, output, self});
        $_state.addAttribute("dim",$_builder.getI64IntegerAttr(dim));
        $_state.addTypes(grad_output.getType());
  }]>];
}

def Poptorch_softplus : Poptorch_Op<"softplus", []> {
  let arguments = (ins Poptorch_tensor:$input,
                       F32Attr:$beta,
                       F32Attr:$threshold);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$input,
                                 "float":$beta,
                                 "float":$threshold), [{
        $_state.addOperands({input});
        $_state.addAttribute("beta", $_builder.getF32FloatAttr(beta));
        $_state.addAttribute("threshold", $_builder.getF32FloatAttr(threshold));
        $_state.addTypes(input.getType());
  }]>];
}

def Poptorch_glu_out : Poptorch_NotImplementedOp<"glu_out", []> {
  let arguments = (ins Poptorch_tensor:$self, I64Attr:$dim);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self, "std::int64_t":$dim), [{
        $_state.addOperands({self});

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        auto ref = tensor.getShape();

        dim = convertToPositiveDim(dim, ref.size());

        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

        llvm::SmallVector<std::int64_t, 4> shape{ref.begin(), ref.end()};

        ERROR_ON_MSG(shape[dim] % 2 != 0,
                     "The size of the given dimension (" + std::to_string(dim) +
                     ") must be a multiple of 2.");
        shape[dim] /= 2;

        $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}
