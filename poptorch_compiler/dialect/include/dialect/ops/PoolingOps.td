// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

def Poptorch_adaptive_avg_pool : Poptorch_Op<"adaptive_avg_pool", []> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$output_size);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input, "const std::vector<std::int64_t>&":$output_size),[{
        $_state.addOperands({input});
        $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
        mlir::RankedTensorType input_type = input.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> input_shape = input_type.getShape();

        ERROR_ON_MSG(input_shape.size() != output_size.size() + 1 &&
                     input_shape.size() != output_size.size() + 2,
                     "The output size (" << output_size.size() <<
                     ") must be 1 or 2 less than the input rank ("
                     << input_shape.size() << ")");

        std::copy(output_size.begin(), output_size.end(), input_shape.end() - output_size.size());
        $_state.addTypes({mlir::RankedTensorType::get(input_shape, input_type.getElementType())});
      }]>
    ];
}


def Poptorch_avg_pool : Poptorch_Op<"avg_pool", []> {
    let arguments = (ins Poptorch_tensor:$input,
                     I64ArrayAttr:$kernel_size,
                     I64ArrayAttr:$stride,
                     I64ArrayAttr:$padding,
                     BoolAttr:$ceil_mode,
                     BoolAttr:$count_include_pad,
                     OptionalAttr<I64Attr>:$divisor_override);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input,
                                      "const std::vector<std::int64_t>&":$kernel_size,
                                      "const std::vector<std::int64_t>&":$stride,
                                      "const std::vector<std::int64_t>&":$padding,
                                      "bool":$ceil_mode,
                                      "bool":$count_include_pad,
                                      "std::optional<std::int64_t>":$divisor_override),[{
        $_state.addOperands({input});
        $_state.addAttribute("kernel_size", $_builder.getI64ArrayAttr(kernel_size));
        $_state.addAttribute("stride", $_builder.getI64ArrayAttr(stride));
        $_state.addAttribute("padding", $_builder.getI64ArrayAttr(padding));
        $_state.addAttribute("ceil_mode", $_builder.getBoolAttr(ceil_mode));
        $_state.addAttribute("count_include_pad", $_builder.getBoolAttr(count_include_pad));
        if (divisor_override.has_value()) {
          $_state.addAttribute("divisor_override", $_builder.getSI32IntegerAttr(divisor_override.value()));
        }

        mlir::RankedTensorType input_type = input.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> input_shape = input_type.getShape();

        ERROR_ON_MSG(input_shape.size() != kernel_size.size() + 1 &&
                     input_shape.size() != kernel_size.size() + 2,
                     "The kernel size (" << kernel_size.size() <<
                     ") must be 1 or 2 less than the input rank ("
                     << input_shape.size() << ")");
        ERROR_ON(kernel_size.size() != stride.size());
        ERROR_ON(kernel_size.size() != padding.size());

        size_t offset = (input_shape.size() == kernel_size.size() + 1) ? 1 : 2;
        for (auto s = 0u; s < kernel_size.size(); s++) {
          double tmp = (input_shape[offset + s] + 2. * padding[s] - kernel_size[s]) / stride[s] + 1.;
          if (ceil_mode) {
            input_shape[offset + s] = std::ceil(tmp);
          } else {
            input_shape[offset + s] = std::floor(tmp);
          }
        }
        $_state.addTypes({mlir::RankedTensorType::get(input_shape, input_type.getElementType())});
      }]>
    ];
}


def Poptorch_max_pool : Poptorch_Op<"max_pool", []> {
    let arguments = (ins Poptorch_tensor:$input,
                     I64ArrayAttr:$kernel_size,
                     I64ArrayAttr:$stride,
                     I64ArrayAttr:$padding,
                     I64ArrayAttr:$dilation,
                     BoolAttr:$ceil_mode);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input,
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

        ERROR_ON_MSG(input_shape.size() != kernel_size.size() + 1 &&
                     input_shape.size() != kernel_size.size() + 2,
                     "The kernel size (" << kernel_size.size() <<
                     ") must be 1 or 2 less than the input rank ("
                     << input_shape.size() << ")");
        ERROR_ON(kernel_size.size() != stride.size());
        ERROR_ON(kernel_size.size() != padding.size());
        ERROR_ON(kernel_size.size() != dilation.size());

        const size_t offset = (input_shape.size() == kernel_size.size() + 1) ? 1 : 2;

        for (auto s = 0u; s < kernel_size.size(); s++) {
          double tmp = (input_shape[offset + s] + 2. * padding[s] - dilation[s] * (kernel_size[s] - 1.) - 1.) / stride[s] + 1.;
          if (ceil_mode) {
            input_shape[offset + s] = std::ceil(tmp);
          } else {
            input_shape[offset + s] = std::floor(tmp);
          }
        }
        $_state.addTypes({mlir::RankedTensorType::get(input_shape, input_type.getElementType())});
      }]>
    ];
}
