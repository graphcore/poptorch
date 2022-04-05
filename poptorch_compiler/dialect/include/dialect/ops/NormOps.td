// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

def Poptorch_batch_norm : Poptorch_Op<"batch_norm", [AttrSizedOperandSegments]> {
  let arguments = (ins Poptorch_tensor:$input,
                        Optional<Poptorch_tensor>:$weight,
                        Optional<Poptorch_tensor>:$bias,
                        Optional<Poptorch_tensor>:$running_mean,
                        Optional<Poptorch_tensor>:$running_var,
                        BoolAttr:$training,
                        F32Attr:$momentum,
                        F32Attr:$epsilon);
  let results = (outs Poptorch_tensor:$result, Poptorch_tensor:$save_mean, Poptorch_tensor:$save_invstd);
  let builders = [OpBuilderDAG<(ins "mlir::Value":$input,
                                    "mlir::Value":$weight,
                                    "mlir::Value":$bias,
                                    "mlir::Value":$running_mean,
                                    "mlir::Value":$running_var,
                                    "bool":$training,
                                    "float":$momentum,
                                    "float":$epsilon), [{
        std::vector<mlir::Value> operands = {input};
        // These segments represent the optional tensors in the order they appear in the arguments.
        // For each segment, 1 indicates a value has been passed for that argument, otherwise 0
        // indicates a nullopt
        // All operands must have a value: 1 for non optional/variadic
        std::vector<std::int32_t> segments = {1, 0, 0, 0, 0};
        if (weight) {
          operands.push_back(weight);
          operands.push_back(bias);
          segments[1] = 1; // weight
          segments[2] = 1; // bias
        }
        if (running_mean) {
          operands.push_back(running_mean);
          operands.push_back(running_var);
          segments[3] = 1; // running_mean
          segments[4] = 1; // running_var
        }
        $_state.addOperands(operands);
        $_state.addAttribute("training",$_builder.getBoolAttr(training));
        $_state.addAttribute("momentum",$_builder.getF32FloatAttr(momentum));
        $_state.addAttribute("epsilon",$_builder.getF32FloatAttr(epsilon));
        $_state.addAttribute("operand_segment_sizes", $_builder.getI32VectorAttr(segments));
        mlir::Type running_type = mlir::RankedTensorType::get({input.getType().cast<mlir::RankedTensorType>().getDimSize(1)}, input.getType().cast<mlir::RankedTensorType>().getElementType());
        $_state.addTypes({input.getType(), running_type, running_type});
  }]>];
}

def Poptorch_group_norm : Poptorch_Op<"group_norm",
                                      [AttrSizedOperandSegments]> {
  let arguments = (ins Poptorch_tensor:$input,
                       Optional<Poptorch_tensor>:$weight,
                       Optional<Poptorch_tensor>:$bias,
                       I64Attr:$N,
                       I64Attr:$C,
                       I64Attr:$HxW,
                       I64Attr:$group,
                       F32Attr:$epsilon);

  let results = (outs Poptorch_tensor:$result, Poptorch_tensor:$mean,
                      Poptorch_tensor:$rstd);

  let builders = [OpBuilderDAG<(ins "mlir::Value":$input,
                                    "mlir::Value":$weight,
                                    "mlir::Value":$bias,
                                    "int":$N,
                                    "int":$C,
                                    "int":$HxW,
                                    "int":$group, //  num_groups (PyTorch op)
                                    "float":$epsilon), [{

        std::vector<mlir::Value> operands = {input};
        // These segments represent the optional tensors in the order they appear in the arguments.
        // For each segment, 1 indicates a value has been passed for that argument, otherwise 0
        // indicates a nullopt
        // All operands must have a value: 1 for non optional/variadic
        std::vector<std::int32_t> segments = {1, 0, 0};
        if (weight) {
          ERROR_ON(!bias);

          operands.push_back(weight);
          operands.push_back(bias);
          segments[1] = 1; // weight
          segments[2] = 1; // bias
        }

        $_state.addOperands(operands);

        $_state.addAttribute("group", $_builder.getI64IntegerAttr(group));
        $_state.addAttribute("epsilon",$_builder.getF32FloatAttr(epsilon));
        $_state.addAttribute("operand_segment_sizes",
                             $_builder.getI32VectorAttr(segments));

        // These are not neeeded but useful as a check when lowering
        $_state.addAttribute("N", $_builder.getI64IntegerAttr(N));
        $_state.addAttribute("C",$_builder.getI64IntegerAttr(C));
        $_state.addAttribute("HxW",$_builder.getI64IntegerAttr(HxW));



        std::vector<int64_t> new_shape{N, group};
        mlir::Type mean_rstd_type = mlir::RankedTensorType::get(
          new_shape, input.getType().cast<mlir::RankedTensorType>().getElementType());

        $_state.addTypes({input.getType(), mean_rstd_type, mean_rstd_type});
  }]>];
}

// aten::native_batch_norm_backward(Tensor grad_out, Tensor input,
//                                  Tensor? weight, Tensor? running_mean,
//                                  Tensor? running_var, Tensor? save_mean,
//                                  Tensor? save_invstd, bool train, float eps,
//                                  bool[3] output_mask) -> (Tensor, Tensor, Tensor)
def Poptorch_batch_norm_backward: Poptorch_Op<"batch_norm_backward", [AttrSizedOperandSegments]> {
  let arguments = (ins Poptorch_tensor:$grad_out,
                       Poptorch_tensor:$input,
                       Poptorch_tensor:$weight,
                       Optional<Poptorch_tensor>:$running_mean,
                       Optional<Poptorch_tensor>:$running_var,
                       Optional<Poptorch_tensor>:$save_mean,
                       Optional<Poptorch_tensor>:$save_invstd,
                       BoolAttr:$training,
                       F32Attr:$epsilon);
  let results = (outs Poptorch_tensor:$grad_input, Poptorch_tensor:$grad_weight, Poptorch_tensor:$grad_bias);
  let builders = [OpBuilderDAG<(ins "mlir::Value":$grad_out,
                                    "mlir::Value":$input,
                                    "mlir::Value":$weight,
                                    "mlir::Value":$running_mean,
                                    "mlir::Value":$running_var,
                                    "mlir::Value":$save_mean,
                                    "mlir::Value":$save_invstd,
                                    "bool":$training,
                                    "float":$epsilon), [{
    // Weight would ordinarily be optional but we always populate it in the
    // forward pass if it doesn't exist
    std::vector<mlir::Value> operands = {grad_out, input, weight};
    // These segments represent the optional tensors in the order they appear in the arguments.
    // For each segment, 1 indicates a value has been passed for that argument, otherwise 0
    // indicates a nullopt
    // All operands must have a value: 1 for non optional/variadic
    std::vector<std::int32_t> segments = {1, 1, 1, 0, 0, 0, 0};
    if (running_mean) {
      ERROR_ON(!running_var);
      operands.push_back(running_mean);
      operands.push_back(running_var);
      segments[3] = 1; // running_mean
      segments[4] = 1; // running_var
    }
    if (save_mean) {
      ERROR_ON(!save_invstd);
      operands.push_back(save_mean);
      operands.push_back(save_invstd);
      segments[5] = 1; // save_mean
      segments[6] = 1; // save_invstd
    }
    $_state.addOperands(operands);
    $_state.addAttribute("training",$_builder.getBoolAttr(training));
    $_state.addAttribute("epsilon",$_builder.getF32FloatAttr(epsilon));
    $_state.addAttribute("operand_segment_sizes", $_builder.getI32VectorAttr(segments));
    $_state.addTypes({input.getType(), weight.getType(), weight.getType()});
  }]>];
}

def Poptorch_layer_norm : Poptorch_NotImplementedOp<"layer_norm", [AttrSizedOperandSegments]> {
  let arguments = (ins Poptorch_tensor:$input,
                        I64ArrayAttr:$normalized_shape,
                        Optional<Poptorch_tensor>:$weight,
                        Optional<Poptorch_tensor>:$bias,
                        F32Attr:$epsilon);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::Value":$input,
                                    "const std::vector<std::int64_t>&":$normalized_shape,
                                    "mlir::Value":$weight,
                                    "mlir::Value":$bias,
                                    "float":$epsilon), [{
    $_state.addOperands({input, weight, bias});
    $_state.addAttribute("normalized_shape",$_builder.getI64ArrayAttr(normalized_shape));
    $_state.addAttribute("epsilon",$_builder.getF32FloatAttr(epsilon));
    $_state.addTypes(input.getType());
  }]>];
}