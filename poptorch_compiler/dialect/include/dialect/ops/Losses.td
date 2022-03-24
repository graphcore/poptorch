// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> (Tensor)
def Poptorch_mse_loss_backward : Poptorch_Op<"mse_loss_backward", []> {
    let arguments = (ins Poptorch_tensor:$grad_output, Poptorch_tensor:$input, Poptorch_tensor:$target, I64Attr:$reduction);
    let results = (outs Poptorch_tensor:$result);


    let builders = [OpBuilderDAG<(ins "mlir::Value":$grad, "mlir::Value":$input, "mlir::Value":$target, "std::int64_t":$reduction),[{
        $_state.addOperands({grad, input, target});

        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addTypes(input.getType());
     }]>
    ];
}

//FIXME(T59038) replace Poptorch_NotImplementedOp -> Poptorch_Op
def Poptorch_mse_loss : Poptorch_NotImplementedOp<"mse_loss", []> {
    let arguments = (ins Poptorch_tensor:$self,
                         Poptorch_tensor:$target,
                         I64Attr:$reduction);

    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "std::int64_t":$reduction),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));
        mlir::RankedTensorType input_type = self.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> shape;
        if (reduction == 0){
            shape = input_type.getShape();
            shape.erase(shape.begin());
        } else {
            shape.push_back(1);
        }

        $_state.addTypes(mlir::RankedTensorType::get(shape, input_type.getElementType()));
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
    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "mlir::Value":$target,
                                      "mlir::Value":$weight,
                                      "std::int64_t":$reduction,
                                      "std::int64_t":$ignore_index),[{
        $_state.addOperands({self, target});
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("ignore_index",$_builder.getI64IntegerAttr(ignore_index));
        mlir::RankedTensorType input_type = self.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> shape;
        if (reduction == 0){
            shape = input_type.getShape();
            shape.erase(shape.begin());
        } else {
            shape.push_back(1);
        }

        $_state.addTypes({mlir::RankedTensorType::get(shape, input_type.getElementType()),
                          mlir::RankedTensorType::get(shape, input_type.getElementType())});
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
    let builders = [OpBuilderDAG<(ins "mlir::Value":$grad_output,
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

