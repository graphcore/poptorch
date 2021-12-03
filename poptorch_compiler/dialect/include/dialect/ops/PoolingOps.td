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


class Poptorch_max_pool<string name> : Poptorch_Op<!strconcat("max_pool", name), []> {
    let arguments = (ins Poptorch_tensor:$input,
                     I64ArrayAttr:$kernel_size,
                     I64ArrayAttr:$stride,
                     I64ArrayAttr:$padding,
                     I64ArrayAttr:$dilation,
                     BoolAttr:$ceil_mode);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$input,
                                      "const std::vector<std::int64_t>&":$kernel_size,
                                      "const std::vector<std::int64_t>&":$stride,
                                      "const std::vector<std::int64_t>&":$padding,
                                      "const std::vector<std::int64_t>&":$dilation,
                                      "bool":$ceil_mode),[{
        $_state.addOperands({input});
        $_state.addAttribute("kernel_size", $_builder.getI64ArrayAttr(kernel_size));
        $_state.addAttribute("stride", $_builder.getI64ArrayAttr(stride));
        $_state.addAttribute("padding", $_builder.getI64ArrayAttr(padding));
        $_state.addAttribute("dilation", $_builder.getI64ArrayAttr(dilation));
        $_state.addAttribute("ceil_mode", $_builder.getBoolAttr(ceil_mode));

        mlir::RankedTensorType input_type = input.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> input_shape = input_type.getShape();

        for (auto s = 0u; s < kernel_size.size(); s++) {
          double tmp = (input_shape[2 + s] + 2. * padding[s] - dilation[s] * (kernel_size[s] - 1.) - 1.) / stride[s] + 1.;
          if (ceil_mode) {
            input_shape[2 + s] = std::ceil(tmp);
          } else {
            input_shape[2 + s] = std::floor(tmp);
          }
        }
        $_state.addTypes({mlir::RankedTensorType::get(input_shape, input_type.getElementType())});
      }]>
    ];
}
def Poptorch_max_pool1d : Poptorch_max_pool<"1d"> {}
def Poptorch_max_pool2d : Poptorch_max_pool<"2d"> {}
def Poptorch_max_pool3d : Poptorch_max_pool<"3d"> {}
