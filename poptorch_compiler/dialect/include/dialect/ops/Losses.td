
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