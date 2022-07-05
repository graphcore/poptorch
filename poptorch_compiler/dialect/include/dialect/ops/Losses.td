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

class Poptorch_not_implemented_simple_reduced_loss<string mnemonic, list<OpTrait> traits = []> :
    Poptorch_NotImplementedOp<mnemonic, traits> {
    let extraClassDeclaration = [{
        static llvm::ArrayRef<std::int64_t> inferShape(const mlir::RankedTensorType &tensor,
                                                       int reduction) {
            return reduction == 0 // ie. 'None'
                ? tensor.getShape()
                : llvm::ArrayRef<std::int64_t> {};
        }

        static llvm::ArrayRef<std::int64_t> inferShape(const mlir::Value &input,
                                                       int reduction) {
            mlir::RankedTensorType tensor = input.getType().cast<mlir::RankedTensorType>();
            return inferShape(tensor, reduction);
        }

        static mlir::RankedTensorType inferType(const mlir::Value &input,
                                                int reduction) {
            mlir::RankedTensorType tensor = input.getType().cast<mlir::RankedTensorType>();
            return mlir::RankedTensorType::get(inferShape(tensor, reduction), tensor.getElementType());
        }
    }];
}

def Poptorch_smooth_l1_loss : Poptorch_not_implemented_simple_reduced_loss<"smooth_l1_loss"> {
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
        $_state.addAttribute("beta", $_builder.getF32FloatAttr(beta));

        $_state.addTypes({inferType(self, reduction)});
    }]>
    ];
}

def Poptorch_l1_loss : Poptorch_not_implemented_simple_reduced_loss<"l1_loss"> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         I64Attr:$reduction);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "mlir::Value":$target,
                                   "std::int64_t":$reduction),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));

        $_state.addTypes({inferType(self, reduction)});
    }]>
    ];
}

def Poptorch_cosine_embedding_loss : Poptorch_not_implemented_simple_reduced_loss<"cosine_embedding_loss"> {
    let arguments = (ins Poptorch_tensor:$input1,
                         Poptorch_tensor:$input2,
                         Poptorch_tensor:$target,
                         F32Attr:$margin,
                         I64Attr:$reduction);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$input1,
                                   "mlir::Value":$input2,
                                   "mlir::Value":$target,
                                   "float":$margin,
                                   "std::int64_t":$reduction),[{
        $_state.addOperands({input1, input2, target});
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("margin", $_builder.getF32FloatAttr(margin));

        const auto shape = inferShape(target, reduction);
        $_state.addTypes({mlir::RankedTensorType::get(shape, $_builder.getF32Type())});
    }]>
    ];
}

def Poptorch_hinge_embedding_loss : Poptorch_not_implemented_simple_reduced_loss<"hinge_embedding_loss"> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         F32Attr:$margin,
                         I64Attr:$reduction);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "mlir::Value":$target,
                                   "float":$margin,
                                   "std::int64_t":$reduction),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("margin", $_builder.getF32FloatAttr(margin));

        $_state.addTypes({inferType(self, reduction)});
    }]>
    ];
}

def Poptorch_kl_div : Poptorch_not_implemented_simple_reduced_loss<"kl_div"> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         I64Attr:$reduction,
                         BoolAttr:$log_target);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "mlir::Value":$target,
                                   "std::int64_t":$reduction,
                                   "bool":$log_target),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("log_target", $_builder.getBoolAttr(log_target));

        $_state.addTypes({inferType(self, reduction)});
    }]>
    ];
}

def Poptorch_margin_ranking_loss : Poptorch_not_implemented_simple_reduced_loss<"margin_ranking_loss"> {
    let arguments = (ins Poptorch_tensor:$input1,
                         Poptorch_tensor:$input2,
                         Poptorch_tensor:$target,
                         F32Attr:$margin,
                         I64Attr:$reduction);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$input1,
                                   "mlir::Value":$input2,
                                   "mlir::Value":$target,
                                   "float":$margin,
                                   "std::int64_t":$reduction),[{
        $_state.addOperands({input1, input2, target});
        $_state.addAttribute("margin", $_builder.getF32FloatAttr(margin));
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));

        $_state.addTypes({inferType(input1, reduction)});
    }]>
    ];
}

def Poptorch_poisson_nll_loss : Poptorch_not_implemented_simple_reduced_loss<"poisson_nll_loss"> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         BoolAttr:$log_input,
                         BoolAttr:$full,
                         F32Attr:$eps,
                         I64Attr:$reduction);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "mlir::Value":$target,
                                   "bool":$log_input,
                                   "bool":$full,
                                   "float":$eps,
                                   "std::int64_t":$reduction),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("log_input", $_builder.getBoolAttr(log_input));
        $_state.addAttribute("full", $_builder.getBoolAttr(full));
        $_state.addAttribute("eps", $_builder.getF32FloatAttr(eps));
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));

        $_state.addTypes({inferType(self, reduction)});
    }]>
    ];
}

def Poptorch_triplet_margin_loss : Poptorch_not_implemented_simple_reduced_loss<"triplet_margin_loss"> {
    let arguments = (ins Poptorch_tensor:$anchor,
                         Poptorch_tensor:$positive,
                         Poptorch_tensor:$negative,
                         F32Attr:$margin,
                         F32Attr:$p,
                         F32Attr:$eps,
                         BoolAttr:$swap,
                         I64Attr:$reduction);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$anchor,
                                   "mlir::Value":$positive,
                                   "mlir::Value":$negative,
                                   "float":$margin,
                                   "float":$p,
                                   "float":$eps,
                                   "bool":$swap,
                                   "std::int64_t":$reduction),[{
        $_state.addOperands({anchor, positive, negative});
        $_state.addAttribute("margin", $_builder.getF32FloatAttr(margin));
        $_state.addAttribute("p", $_builder.getF32FloatAttr(p));
        $_state.addAttribute("eps", $_builder.getF32FloatAttr(eps));
        $_state.addAttribute("swap", $_builder.getBoolAttr(swap));
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));

        mlir::RankedTensorType tensor = anchor.getType().cast<mlir::RankedTensorType>();
        llvm::ArrayRef<std::int64_t> shape;
        if (reduction == 0 && tensor.getShape().size() > 1) {
          shape = tensor.getShape().take_front();
        }

        $_state.addTypes({mlir::RankedTensorType::get(shape, tensor.getElementType())});
    }]>
    ];
}

def Poptorch_ctc_loss_intlist : Poptorch_not_implemented_simple_reduced_loss<"ctc_loss_intlist"> {
    let arguments = (ins Poptorch_tensor:$log_probs,
                         Poptorch_tensor:$targets,
                         I64ArrayAttr:$input_lengths,
                         I64ArrayAttr:$target_lengths,
                         I64Attr:$blank,
                         I64Attr:$reduction,
                         BoolAttr:$zero_infinity);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$log_probs,
                                   "mlir::Value":$targets,
                                   "std::vector<std::int64_t>":$input_lengths,
                                   "std::vector<std::int64_t>":$target_lengths,
                                   "std::int64_t":$blank,
                                   "std::int64_t":$reduction,
                                   "bool":$zero_infinity),[{
        $_state.addOperands({log_probs, targets});
        $_state.addAttribute("input_lengths", $_builder.getI64ArrayAttr(input_lengths));
        $_state.addAttribute("target_lengths", $_builder.getI64ArrayAttr(target_lengths));
        $_state.addAttribute("blank", $_builder.getI64IntegerAttr(blank));
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("zero_infinity", $_builder.getBoolAttr(zero_infinity));

        llvm::ArrayRef<std::int64_t> shape;
        if (reduction == 0) {
          shape = input_lengths;
        }

        $_state.addTypes({mlir::RankedTensorType::get(shape, getElementType(log_probs))});
    }]>
    ];
}

def Poptorch_ctc_loss_tensor : Poptorch_not_implemented_simple_reduced_loss<"ctc_loss_tensor"> {
    let arguments = (ins Poptorch_tensor:$log_probs,
                         Poptorch_tensor:$targets,
                         Poptorch_tensor:$input_lengths,
                         Poptorch_tensor:$target_lengths,
                         I64Attr:$blank,
                         I64Attr:$reduction,
                         BoolAttr:$zero_infinity);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$log_probs,
                                   "mlir::Value":$targets,
                                   "mlir::Value":$input_lengths,
                                   "mlir::Value":$target_lengths,
                                   "std::int64_t":$blank,
                                   "std::int64_t":$reduction,
                                   "bool":$zero_infinity),[{
        $_state.addOperands({log_probs, input_lengths, target_lengths, targets});
        $_state.addAttribute("blank", $_builder.getI64IntegerAttr(blank));
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("zero_infinity", $_builder.getBoolAttr(zero_infinity));

        // If no reduction, get the batch size; from docs, this will be
        // `log_probs`' second dimension if it's 3D.
        mlir::RankedTensorType tensor = log_probs.getType().cast<mlir::RankedTensorType>();
        llvm::SmallVector<std::int64_t, 1> shape;
        if (reduction == 0 && tensor.getShape().size() == 3) {
          shape = {tensor.getShape()[1]};
        }

        $_state.addTypes({mlir::RankedTensorType::get(shape, getElementType(log_probs))});
    }]>
    ];
}