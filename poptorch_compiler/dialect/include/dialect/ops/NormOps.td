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
  let results = (outs Poptorch_tensor:$result, Poptorch_tensor:$mean, Poptorch_tensor:$var);
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
