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


// Inplace versions.
def Poptorch_hardsigmoid_: Poptorch_elem_unary_in_place<"hardsigmoid_"> {}
def Poptorch_hardswish_: Poptorch_elem_unary_in_place<"hardswish_"> {}
def Poptorch_swish_: Poptorch_elem_unary_in_place<"swish_"> {}
def Poptorch_relu_: Poptorch_elem_unary_in_place<"relu_"> {}
def Poptorch_gelu_: Poptorch_elem_unary_in_place<"gelu_"> {}



// Softmax takes an additional "dim" argument.
def Poptorch_softmax : Poptorch_Op<"softmax", [SameOperandsAndResultShape]> {

  // For now we just ignore the float to half bit.
  let arguments = (ins Poptorch_tensor:$input, I64Attr:$axis, BoolAttr:$half_to_float);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilderDAG<(ins "mlir::Value":$input, "std::int64_t":$axis, "bool":$half_to_float), [{
        $_state.addOperands({input});
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));
        $_state.addAttribute("half_to_float",$_builder.getBoolAttr(half_to_float));
        $_state.addTypes(input.getType());
  }]>];
}


def Poptorch_logsoftmax : Poptorch_Op<"logsoftmax", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim, BoolAttr:$half_to_float);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::Value":$input, "std::int64_t":$dim, "bool":$half_to_float), [{
        $_state.addOperands({input});
        $_state.addAttribute("dim",$_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("half_to_float",$_builder.getBoolAttr(half_to_float));
        $_state.addTypes(input.getType());
  }]>];
}


def Poptorch_logsoftmax_backward : Poptorch_Op<"logsoftmax_backward", [SameOperandsAndResultShape]> {
  let arguments = (ins Poptorch_tensor:$grad_output, Poptorch_tensor:$output, I64Attr:$dim, Poptorch_tensor:$self);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::Value":$grad_output, "mlir::Value":$output, "std::int64_t":$dim, "mlir::Value":$self), [{
        $_state.addOperands({grad_output, output, self});
        $_state.addAttribute("dim",$_builder.getI64IntegerAttr(dim));
        $_state.addTypes(grad_output.getType());
  }]>];
}

def Poptorch_softplus : Poptorch_NotImplementedOp<"softplus", []> {
  let arguments = (ins Poptorch_tensor:$input);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::Value":$input), [{
        $_state.addOperands({input});
        $_state.addTypes(input.getType());
  }]>];

}
