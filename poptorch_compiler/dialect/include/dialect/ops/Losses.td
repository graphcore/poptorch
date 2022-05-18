// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

def Poptorch_mse_loss : Poptorch_Op<"mse_loss", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       Poptorch_tensor:$target,
                       I64Attr:$reduction);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "mlir::Value":$target,
                                    "std::int64_t":$reduction),[{
      $_state.addOperands({self, target});
      $_state.addAttribute(
          "reduction", $_builder.getI64IntegerAttr(reduction));
      mlir::RankedTensorType tensor =
          self.getType().cast<mlir::RankedTensorType>();
      std::vector<std::int64_t> shape;

      // If reduction is none, the shape is equal to the input.
      if (reduction == 0) {
        shape = tensor.getShape();
      }
      // Otherwise reduce to a scalar by passing an empty shape
      // This corresponds to torch.Size([])

      $_state.addTypes(mlir::RankedTensorType::get(
          shape, tensor.getElementType()));
    }]>
    ];
}

// aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int
// reduction) -> (Tensor)
def Poptorch_mse_loss_backward : Poptorch_Op<"mse_loss_backward", []> {
  let arguments = (ins Poptorch_tensor:$grad_output,
                       Poptorch_tensor:$input,
                       Poptorch_tensor:$target,
                       I64Attr:$reduction);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$grad,
                                    "mlir::Value":$input,
                                    "mlir::Value":$target,
                                    "std::int64_t":$reduction),[{
      $_state.addOperands({grad, input, target});

      $_state.addAttribute(
          "reduction", $_builder.getI64IntegerAttr(reduction));
      $_state.addTypes(input.getType());
    }]>
    ];
}

def Poptorch_nll_loss : Poptorch_Op<"nll_loss", []> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         Optional<Poptorch_tensor>:$weight,
                         I64Attr:$reduction,
                         I64Attr:$ignore_index);

    let results = (outs Poptorch_tensor:$result,
                        Poptorch_tensor:$total_weight);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "mlir::Value":$weight,
                                      "std::int64_t":$reduction,
                                      "std::int64_t":$ignore_index),[{
        mlir::RankedTensorType target_t = target.getType().cast<
                                          mlir::RankedTensorType>();
        ERROR_ON_MSG(!target_t.getElementType().isIntOrIndex(),
                     "An NLLLoss target must be an integer type.");

        $_state.addOperands({self, target});
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("ignore_index",$_builder.getI64IntegerAttr(ignore_index));
        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> shape;
        // If reduction is none, the shape is the the input without number of
        // classes, which is the second element, i.e. (N, C, ...) to (N, ...)
        // except in the case of a 1D input (C) when it is ().
        if (reduction == 0){
            shape = tensor.getShape();
            if(shape.size() == 1) {
              shape.clear();

            } else {
              ERROR_ON(shape.size() < 2);
              shape.erase(shape.begin() + 1);
            }
        }

        $_state.addTypes({mlir::RankedTensorType::get(shape, tensor.getElementType()),
                          mlir::RankedTensorType::get({}, tensor.getElementType())});
    }]>
    ];
}

def Poptorch_nll_loss_backward : Poptorch_Op<"nll_loss_backward", []> {
    let arguments = (ins Poptorch_tensor:$grad_output,
                         Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         Optional<Poptorch_tensor>:$weight,
                         I64Attr:$reduction,
                         I64Attr:$ignore_index,
                         Poptorch_tensor:$total_weight);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$grad_output,
                                      "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "mlir::Value":$weight,
                                      "std::int64_t":$reduction,
                                      "std::int64_t":$ignore_index,
                                      "mlir::Value":$total_weight),[{
        $_state.addOperands({grad_output, self, target, total_weight});
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("ignore_index",$_builder.getI64IntegerAttr(ignore_index));
        $_state.addTypes(self.getType());
    }]>
    ];

}

def Poptorch_binary_cross_entropy : Poptorch_Op<"binary_cross_entropy", []> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         Optional<Poptorch_tensor>:$weight,
                         I64Attr:$reduction);
  
    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "mlir::Value":$weight,
                                      "std::int64_t":$reduction),[{
        std::vector<mlir::Value> operands = {self, target};

        if(weight) {
          // Weight will have been expanded at this point to the broadcasted
          //shape of weight and target.
          operands.push_back(weight);
        }
        $_state.addOperands(operands);

        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));

        std::vector<std::int64_t> shape;
        if (reduction == 0){
            shape = getShape(self);

            // Note that weight can be broadcast to shape, but shape must not
            // change. This should have been be checked by PyTorch by known
            // currently supported dispatch paths, so there is no extra error
            // message here.
            if(weight) {
              ERROR_ON(shape != broadcast(shape, getShape(weight)));
            }
        }

        $_state.addTypes({mlir::RankedTensorType::get(shape, getElementType(self))});
    }]>
    ];
}

def Poptorch_binary_cross_entropy_backward : Poptorch_Op<"binary_cross_entropy_backward", []> {
    let arguments = (ins Poptorch_tensor:$grad_output,
                         Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         Optional<Poptorch_tensor>:$weight,
                         I64Attr:$reduction);
  
    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$grad_output,
                                      "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "mlir::Value":$weight,
                                      "std::int64_t":$reduction),[{
        std::vector<mlir::Value> operands = {grad_output, self, target};

        if(weight) {
          // Weight will have been expanded at this point to the broadcasted
          // shape of weight and target.
          operands.push_back(weight);
        }
        $_state.addOperands(operands);

        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addTypes(self.getType());
    }]>
    ];
}

def Poptorch_binary_cross_entropy_with_logits : Poptorch_Op<"binary_cross_entropy_with_logits", [SameVariadicOperandSize]> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         Optional<Poptorch_tensor>:$weight,
                         Optional<Poptorch_tensor>:$pos_weight,
                         I64Attr:$reduction);
  
    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "mlir::Value":$weight,
                                      "mlir::Value":$pos_weight,
                                      "std::int64_t":$reduction),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> shape;
        if (reduction == 0){  // if none reduction applied the output shape is the same as input, otherwise torch.Size([])
            shape = tensor.getShape();
        }
        $_state.addTypes({mlir::RankedTensorType::get(shape, tensor.getElementType())});
    }]>
    ];
}

def Poptorch_binary_cross_entropy_with_logits_backward : Poptorch_Op<"binary_cross_entropy_with_logits_backward", [SameVariadicOperandSize]> {
    let arguments = (ins Poptorch_tensor:$grad_output,
                         Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         Optional<Poptorch_tensor>:$weight,
                         Optional<Poptorch_tensor>:$pos_weight,
                         I64Attr:$reduction);
  
    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$grad_output,
                                      "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "mlir::Value":$weight,
                                      "mlir::Value":$pos_weight,
                                      "std::int64_t":$reduction),[{
        $_state.addOperands({grad_output, self, target});
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addTypes(self.getType());
    }]>
    ];
}

def Poptorch_smooth_l1_loss : Poptorch_NotImplementedOp<"smooth_l1_loss", []> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         I64Attr:$reduction,
                         F32Attr:$beta);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "std::int64_t":$reduction,
                                      "float":$beta),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("beta",$_builder.getF32FloatAttr(beta));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        llvm::ArrayRef<std::int64_t> shape;
        if (reduction == 0) {  // if none reduction applied the output shape is the same as input, otherwise torch.Size([])
            shape = tensor.getShape();
        }

        $_state.addTypes({mlir::RankedTensorType::get(shape, tensor.getElementType())});
    }]>
    ];
}