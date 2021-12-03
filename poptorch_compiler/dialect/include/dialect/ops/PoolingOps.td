// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

class Poptorch_adaptive_avg_pool<string name> : Poptorch_Op<!strconcat("adaptive_avg_pool", name), []> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$output_size);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$input, "const std::vector<std::int64_t>&":$output_size),[{
        $_state.addOperands({input});
        $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
        mlir::RankedTensorType input_type = input.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> input_shape = input_type.getShape();
        for (auto spatial_dim = 0; spatial_dim < output_size.size(); spatial_dim++) {
          input_shape[2 + spatial_dim] = output_size[spatial_dim];
        }
        $_state.addTypes({mlir::RankedTensorType::get(input_shape, input_type.getElementType())});
      }]>
    ];
}
def Poptorch_adaptive_avg_pool1d : Poptorch_adaptive_avg_pool<"1d"> {}
def Poptorch_adaptive_avg_pool2d : Poptorch_adaptive_avg_pool<"2d"> {}
def Poptorch_adaptive_avg_pool3d : Poptorch_adaptive_avg_pool<"3d"> {}
