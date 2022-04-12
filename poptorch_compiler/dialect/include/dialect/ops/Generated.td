// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

def Poptorch_groupnormalization : Poptorch_Op<"groupnormalization", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$num_groups, F32Attr:$epsilon);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$num_groups, "float":$epsilon), [{

        $_state.addOperands(values);
        $_state.addAttribute("num_groups",$_builder.getI64IntegerAttr(num_groups));
        $_state.addAttribute("epsilon",$_builder.getF32FloatAttr(epsilon));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_subsample : Poptorch_Op<"subsample", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_nop : Poptorch_Op<"nop", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_lstm : Poptorch_Op<"lstm", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$outputFullSequence);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$outputFullSequence), [{

        $_state.addOperands(values);
        $_state.addAttribute("outputFullSequence",$_builder.getI64IntegerAttr(outputFullSequence));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_depthtospace : Poptorch_Op<"depthtospace", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$blocksize, StrAttr:$mode);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$blocksize, "const char*":$mode), [{

        $_state.addOperands(values);
        $_state.addAttribute("blocksize",$_builder.getI64IntegerAttr(blocksize));
        $_state.addAttribute("mode",$_builder.getStringAttr(mode));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_dynamicslice : Poptorch_Op<"dynamicslice", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64ArrayAttr:$sizes, I64Attr:$noOverlap);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "const std::vector<std::int64_t>&":$sizes, "std::int64_t":$noOverlap), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("sizes",$_builder.getI64ArrayAttr(sizes));
        $_state.addAttribute("noOverlap",$_builder.getI64IntegerAttr(noOverlap));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_dynamicupdate : Poptorch_Op<"dynamicupdate", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64ArrayAttr:$sizes, I64Attr:$noOverlap);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "const std::vector<std::int64_t>&":$sizes, "std::int64_t":$noOverlap), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("sizes",$_builder.getI64ArrayAttr(sizes));
        $_state.addAttribute("noOverlap",$_builder.getI64IntegerAttr(noOverlap));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_dynamiczero : Poptorch_Op<"dynamiczero", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64ArrayAttr:$sizes);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "const std::vector<std::int64_t>&":$sizes), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("sizes",$_builder.getI64ArrayAttr(sizes));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_dynamicadd : Poptorch_Op<"dynamicadd", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64ArrayAttr:$sizes);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "const std::vector<std::int64_t>&":$sizes), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("sizes",$_builder.getI64ArrayAttr(sizes));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_sequenceslice : Poptorch_Op<"sequenceslice", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$zeroUnused);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$zeroUnused), [{

        $_state.addOperands(values);
        $_state.addAttribute("zeroUnused",$_builder.getI64IntegerAttr(zeroUnused));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_replicatedallreduce : Poptorch_Op<"replicatedallreduce", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$commGroup);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$commGroup), [{

        $_state.addOperands(values);
        $_state.addAttribute("commGroup",$_builder.getI64ArrayAttr(commGroup));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_l1loss : Poptorch_Op<"l1loss", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, F32Attr:$lambda, I64Attr:$reduction);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "float":$lambda, "std::int64_t":$reduction), [{

        $_state.addOperands(values);
        $_state.addAttribute("lambda",$_builder.getF32FloatAttr(lambda));
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch__ctcloss : Poptorch_Op<"_ctcloss", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$reduction, I64Attr:$blank, StrAttr:$outDataType);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$reduction, "std::int64_t":$blank, "const char*":$outDataType), [{

        $_state.addOperands(values);
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));
        $_state.addAttribute("blank",$_builder.getI64IntegerAttr(blank));
        $_state.addAttribute("outDataType",$_builder.getStringAttr(outDataType));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_ctcbeamsearchdecoder : Poptorch_Op<"ctcbeamsearchdecoder", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$blank, I64Attr:$beamWidth, I64Attr:$topPaths);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$blank, "std::int64_t":$beamWidth, "std::int64_t":$topPaths), [{

        $_state.addOperands(values);
        $_state.addAttribute("blank",$_builder.getI64IntegerAttr(blank));
        $_state.addAttribute("beamWidth",$_builder.getI64IntegerAttr(beamWidth));
        $_state.addAttribute("topPaths",$_builder.getI64IntegerAttr(topPaths));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_shapeddropout : Poptorch_Op<"shapeddropout", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$shape, F32Attr:$ratio);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$shape, "float":$ratio), [{

        $_state.addOperands(values);
        $_state.addAttribute("shape",$_builder.getI64ArrayAttr(shape));
        $_state.addAttribute("ratio",$_builder.getF32FloatAttr(ratio));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reverse : Poptorch_Op<"reverse", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$dimensions);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$dimensions), [{

        $_state.addOperands(values);
        $_state.addAttribute("dimensions",$_builder.getI64ArrayAttr(dimensions));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reducemedian : Poptorch_Op<"reducemedian", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_scatterreduce : Poptorch_Op<"scatterreduce", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis_size, I64Attr:$axis, I64Attr:$reduction);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis_size, "std::int64_t":$axis, "std::int64_t":$reduction), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis_size",$_builder.getI64IntegerAttr(axis_size));
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));
        $_state.addAttribute("reduction",$_builder.getI64IntegerAttr(reduction));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_averagepool : Poptorch_Op<"averagepool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$kernel_shape, I64Attr:$ceil_mode, I64Attr:$count_include_pad, I64ArrayAttr:$pads, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$kernel_shape, "std::int64_t":$ceil_mode, "std::int64_t":$count_include_pad, "const std::vector<std::int64_t>&":$pads, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("kernel_shape",$_builder.getI64ArrayAttr(kernel_shape));
        $_state.addAttribute("ceil_mode",$_builder.getI64IntegerAttr(ceil_mode));
        $_state.addAttribute("count_include_pad",$_builder.getI64IntegerAttr(count_include_pad));
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_convinteger : Poptorch_Op<"convinteger", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$dilations, I64Attr:$group, I64ArrayAttr:$kernel_shape, I64ArrayAttr:$pads, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$dilations, "std::int64_t":$group, "const std::vector<std::int64_t>&":$kernel_shape, "const std::vector<std::int64_t>&":$pads, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("dilations",$_builder.getI64ArrayAttr(dilations));
        $_state.addAttribute("group",$_builder.getI64IntegerAttr(group));
        $_state.addAttribute("kernel_shape",$_builder.getI64ArrayAttr(kernel_shape));
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_dequantizelinear : Poptorch_Op<"dequantizelinear", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_isinf : Poptorch_Op<"isinf", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$detect_negative, I64Attr:$detect_positive);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$detect_negative, "std::int64_t":$detect_positive), [{

        $_state.addOperands(values);
        $_state.addAttribute("detect_negative",$_builder.getI64IntegerAttr(detect_negative));
        $_state.addAttribute("detect_positive",$_builder.getI64IntegerAttr(detect_positive));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_matmulinteger : Poptorch_Op<"matmulinteger", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_maxpool : Poptorch_Op<"maxpool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$num_outputs, I64ArrayAttr:$kernel_shape, I64Attr:$ceil_mode, I64ArrayAttr:$dilations, I64ArrayAttr:$pads, I64Attr:$storage_order, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$num_outputs, "const std::vector<std::int64_t>&":$kernel_shape, "std::int64_t":$ceil_mode, "const std::vector<std::int64_t>&":$dilations, "const std::vector<std::int64_t>&":$pads, "std::int64_t":$storage_order, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("num_outputs",$_builder.getI64IntegerAttr(num_outputs));
        $_state.addAttribute("kernel_shape",$_builder.getI64ArrayAttr(kernel_shape));
        $_state.addAttribute("ceil_mode",$_builder.getI64IntegerAttr(ceil_mode));
        $_state.addAttribute("dilations",$_builder.getI64ArrayAttr(dilations));
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("storage_order",$_builder.getI64IntegerAttr(storage_order));
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_mod : Poptorch_Op<"mod", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$fmod);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$fmod), [{

        $_state.addOperands(values);
        $_state.addAttribute("fmod",$_builder.getI64IntegerAttr(fmod));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_nonmaxsuppression : Poptorch_Op<"nonmaxsuppression", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$center_point_box);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$center_point_box), [{

        $_state.addOperands(values);
        $_state.addAttribute("center_point_box",$_builder.getI64IntegerAttr(center_point_box));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_qlinearconv : Poptorch_Op<"qlinearconv", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$dilations, I64Attr:$group, I64ArrayAttr:$kernel_shape, I64ArrayAttr:$pads, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$dilations, "std::int64_t":$group, "const std::vector<std::int64_t>&":$kernel_shape, "const std::vector<std::int64_t>&":$pads, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("dilations",$_builder.getI64ArrayAttr(dilations));
        $_state.addAttribute("group",$_builder.getI64IntegerAttr(group));
        $_state.addAttribute("kernel_shape",$_builder.getI64ArrayAttr(kernel_shape));
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_qlinearmatmul : Poptorch_Op<"qlinearmatmul", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_quantizelinear : Poptorch_Op<"quantizelinear", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_resize : Poptorch_Op<"resize", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, StrAttr:$mode);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const char*":$mode), [{

        $_state.addOperands(values);
        $_state.addAttribute("mode",$_builder.getStringAttr(mode));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reversesequence : Poptorch_Op<"reversesequence", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$batch_axis, I64Attr:$time_axis);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$batch_axis, "std::int64_t":$time_axis), [{

        $_state.addOperands(values);
        $_state.addAttribute("batch_axis",$_builder.getI64IntegerAttr(batch_axis));
        $_state.addAttribute("time_axis",$_builder.getI64IntegerAttr(time_axis));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_roialign : Poptorch_Op<"roialign", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, StrAttr:$mode, I64Attr:$output_height, I64Attr:$output_width, I64Attr:$sampling_ratio, F32Attr:$spatial_scale);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const char*":$mode, "std::int64_t":$output_height, "std::int64_t":$output_width, "std::int64_t":$sampling_ratio, "float":$spatial_scale), [{

        $_state.addOperands(values);
        $_state.addAttribute("mode",$_builder.getStringAttr(mode));
        $_state.addAttribute("output_height",$_builder.getI64IntegerAttr(output_height));
        $_state.addAttribute("output_width",$_builder.getI64IntegerAttr(output_width));
        $_state.addAttribute("sampling_ratio",$_builder.getI64IntegerAttr(sampling_ratio));
        $_state.addAttribute("spatial_scale",$_builder.getF32FloatAttr(spatial_scale));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_slice : Poptorch_Op<"slice", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_thresholdedrelu : Poptorch_Op<"thresholdedrelu", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, F32Attr:$alpha);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "float":$alpha), [{

        $_state.addOperands(values);
        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_upsample : Poptorch_Op<"upsample", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, StrAttr:$mode);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const char*":$mode), [{

        $_state.addOperands(values);
        $_state.addAttribute("mode",$_builder.getStringAttr(mode));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_acosh : Poptorch_Op<"acosh", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_asinh : Poptorch_Op<"asinh", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_atanh : Poptorch_Op<"atanh", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_compress : Poptorch_Op<"compress", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_cosh : Poptorch_Op<"cosh", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_eyelike : Poptorch_Op<"eyelike", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$dtype, I64Attr:$k);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$dtype, "std::int64_t":$k), [{

        $_state.addOperands(values);
        $_state.addAttribute("dtype",$_builder.getI64IntegerAttr(dtype));
        $_state.addAttribute("k",$_builder.getI64IntegerAttr(k));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_flatten : Poptorch_Op<"flatten", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_gemm : Poptorch_Op<"gemm", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, F32Attr:$alpha, F32Attr:$beta, I64Attr:$transA, I64Attr:$transB);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "float":$alpha, "float":$beta, "std::int64_t":$transA, "std::int64_t":$transB), [{

        $_state.addOperands(values);
        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));
        $_state.addAttribute("beta",$_builder.getF32FloatAttr(beta));
        $_state.addAttribute("transA",$_builder.getI64IntegerAttr(transA));
        $_state.addAttribute("transB",$_builder.getI64IntegerAttr(transB));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_maxunpool : Poptorch_Op<"maxunpool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$kernel_shape, I64ArrayAttr:$pads, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$kernel_shape, "const std::vector<std::int64_t>&":$pads, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("kernel_shape",$_builder.getI64ArrayAttr(kernel_shape));
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_meanvariancenormalization : Poptorch_Op<"meanvariancenormalization", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_nonzero : Poptorch_Op<"nonzero", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];
}

def Poptorch_onehot : Poptorch_Op<"onehot", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_prelu : Poptorch_Op<"prelu", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_scatter : Poptorch_Op<"scatter", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_shrink : Poptorch_Op<"shrink", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, F32Attr:$bias, F32Attr:$lambd);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "float":$bias, "float":$lambd), [{

        $_state.addOperands(values);
        $_state.addAttribute("bias",$_builder.getF32FloatAttr(bias));
        $_state.addAttribute("lambd",$_builder.getF32FloatAttr(lambd));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_sign : Poptorch_Op<"sign", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_sinh : Poptorch_Op<"sinh", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_tfidfvectorizer : Poptorch_Op<"tfidfvectorizer", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$max_gram_length, I64Attr:$max_skip_count, I64Attr:$min_gram_length, StrAttr:$mode, I64ArrayAttr:$ngram_counts, I64ArrayAttr:$ngram_indexes, I64ArrayAttr:$pool_int64s, StrArrayAttr:$pool_strings, F32ArrayAttr:$weights);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$max_gram_length, "std::int64_t":$max_skip_count, "std::int64_t":$min_gram_length, "const char*":$mode, "const std::vector<std::int64_t>&":$ngram_counts, "const std::vector<std::int64_t>&":$ngram_indexes, "const std::vector<std::int64_t>&":$pool_int64s, "const std::vector<const char*>&":$pool_strings, "const std::vector<float>&":$weights), [{

        $_state.addOperands(values);
        $_state.addAttribute("max_gram_length",$_builder.getI64IntegerAttr(max_gram_length));
        $_state.addAttribute("max_skip_count",$_builder.getI64IntegerAttr(max_skip_count));
        $_state.addAttribute("min_gram_length",$_builder.getI64IntegerAttr(min_gram_length));
        $_state.addAttribute("mode",$_builder.getStringAttr(mode));
        $_state.addAttribute("ngram_counts",$_builder.getI64ArrayAttr(ngram_counts));
        $_state.addAttribute("ngram_indexes",$_builder.getI64ArrayAttr(ngram_indexes));
        $_state.addAttribute("pool_int64s",$_builder.getI64ArrayAttr(pool_int64s));
        $_state.addAttribute("pool_strings",$_builder.getStrArrayAttr(convert(pool_strings)));
        $_state.addAttribute("weights",$_builder.getF32ArrayAttr(weights));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_mean : Poptorch_Op<"mean", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_sum : Poptorch_Op<"sum", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_acos : Poptorch_Op<"acos", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_atan : Poptorch_Op<"atan", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_multinomial : Poptorch_Op<"multinomial", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$dtype, I64Attr:$sample_size, F32Attr:$seed);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$dtype, "std::int64_t":$sample_size, "float":$seed), [{

        $_state.addOperands(values);
        $_state.addAttribute("dtype",$_builder.getI64IntegerAttr(dtype));
        $_state.addAttribute("sample_size",$_builder.getI64IntegerAttr(sample_size));
        $_state.addAttribute("seed",$_builder.getF32FloatAttr(seed));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_rnn : Poptorch_Op<"rnn", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$num_outputs, F32ArrayAttr:$activation_alpha, F32ArrayAttr:$activation_beta, StrArrayAttr:$activations, F32Attr:$clip, StrAttr:$direction, I64Attr:$hidden_size);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$num_outputs, "const std::vector<float>&":$activation_alpha, "const std::vector<float>&":$activation_beta, "const std::vector<const char*>&":$activations, "float":$clip, "const char*":$direction, "std::int64_t":$hidden_size), [{

        $_state.addOperands(values);
        $_state.addAttribute("num_outputs",$_builder.getI64IntegerAttr(num_outputs));
        $_state.addAttribute("activation_alpha",$_builder.getF32ArrayAttr(activation_alpha));
        $_state.addAttribute("activation_beta",$_builder.getF32ArrayAttr(activation_beta));
        $_state.addAttribute("activations",$_builder.getStrArrayAttr(convert(activations)));
        $_state.addAttribute("clip",$_builder.getF32FloatAttr(clip));
        $_state.addAttribute("direction",$_builder.getStringAttr(direction));
        $_state.addAttribute("hidden_size",$_builder.getI64IntegerAttr(hidden_size));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_logical_xor : Poptorch_Op<"logical_xor", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}


def Poptorch_clip : Poptorch_Op<"clip", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, F32Attr:$max, F32Attr:$min);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "float":$max, "float":$min), [{

        $_state.addOperands(values);
        $_state.addAttribute("max",$_builder.getF32FloatAttr(max));
        $_state.addAttribute("min",$_builder.getF32FloatAttr(min));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_convtranspose : Poptorch_Op<"convtranspose", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$dilations, I64Attr:$group, I64ArrayAttr:$kernel_shape, I64ArrayAttr:$output_padding, I64ArrayAttr:$output_shape, I64ArrayAttr:$pads, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$dilations, "std::int64_t":$group, "const std::vector<std::int64_t>&":$kernel_shape, "const std::vector<std::int64_t>&":$output_padding, "const std::vector<std::int64_t>&":$output_shape, "const std::vector<std::int64_t>&":$pads, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("dilations",$_builder.getI64ArrayAttr(dilations));
        $_state.addAttribute("group",$_builder.getI64IntegerAttr(group));
        $_state.addAttribute("kernel_shape",$_builder.getI64ArrayAttr(kernel_shape));
        $_state.addAttribute("output_padding",$_builder.getI64ArrayAttr(output_padding));
        $_state.addAttribute("output_shape",$_builder.getI64ArrayAttr(output_shape));
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_gather : Poptorch_Op<"gather", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_globalaveragepool : Poptorch_Op<"globalaveragepool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_globallppool : Poptorch_Op<"globallppool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$p);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$p), [{

        $_state.addOperands(values);
        $_state.addAttribute("p",$_builder.getI64IntegerAttr(p));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_globalmaxpool : Poptorch_Op<"globalmaxpool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_hardmax : Poptorch_Op<"hardmax", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_identity : Poptorch_Op<"identity", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_instancenormalization : Poptorch_Op<"instancenormalization", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, F32Attr:$epsilon);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "float":$epsilon), [{

        $_state.addOperands(values);
        $_state.addAttribute("epsilon",$_builder.getF32FloatAttr(epsilon));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];
}

def Poptorch_lrn : Poptorch_Op<"lrn", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$size, F32Attr:$alpha, F32Attr:$beta, F32Attr:$bias);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$size, "float":$alpha, "float":$beta, "float":$bias), [{

        $_state.addOperands(values);
        $_state.addAttribute("size",$_builder.getI64IntegerAttr(size));
        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));
        $_state.addAttribute("beta",$_builder.getF32FloatAttr(beta));
        $_state.addAttribute("bias",$_builder.getF32FloatAttr(bias));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_lpnormalization : Poptorch_Op<"lpnormalization", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$axis, I64Attr:$p);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$axis, "std::int64_t":$p), [{

        $_state.addOperands(values);
        $_state.addAttribute("axis",$_builder.getI64IntegerAttr(axis));
        $_state.addAttribute("p",$_builder.getI64IntegerAttr(p));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_lppool : Poptorch_Op<"lppool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$kernel_shape, I64Attr:$p, I64ArrayAttr:$pads, I64ArrayAttr:$strides);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$kernel_shape, "std::int64_t":$p, "const std::vector<std::int64_t>&":$pads, "const std::vector<std::int64_t>&":$strides), [{

        $_state.addOperands(values);
        $_state.addAttribute("kernel_shape",$_builder.getI64ArrayAttr(kernel_shape));
        $_state.addAttribute("p",$_builder.getI64IntegerAttr(p));
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("strides",$_builder.getI64ArrayAttr(strides));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_maxroipool : Poptorch_Op<"maxroipool", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$pooled_shape, F32Attr:$spatial_scale);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$pooled_shape, "float":$spatial_scale), [{

        $_state.addOperands(values);
        $_state.addAttribute("pooled_shape",$_builder.getI64ArrayAttr(pooled_shape));
        $_state.addAttribute("spatial_scale",$_builder.getF32FloatAttr(spatial_scale));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_pad : Poptorch_Op<"pad", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$pads, StrAttr:$mode, F32Attr:$value);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$pads, "const char*":$mode, "float":$value), [{

        $_state.addOperands(values);
        $_state.addAttribute("pads",$_builder.getI64ArrayAttr(pads));
        $_state.addAttribute("mode",$_builder.getStringAttr(mode));
        $_state.addAttribute("value",$_builder.getF32FloatAttr(value));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_randomnormallike : Poptorch_Op<"randomnormallike", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$dtype, F32Attr:$mean, F32Attr:$scale, F32Attr:$seed);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$dtype, "float":$mean, "float":$scale, "float":$seed), [{

        $_state.addOperands(values);
        $_state.addAttribute("dtype",$_builder.getI64IntegerAttr(dtype));
        $_state.addAttribute("mean",$_builder.getF32FloatAttr(mean));
        $_state.addAttribute("scale",$_builder.getF32FloatAttr(scale));
        $_state.addAttribute("seed",$_builder.getF32FloatAttr(seed));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_randomuniformlike : Poptorch_Op<"randomuniformlike", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$dtype, F32Attr:$high, F32Attr:$low, F32Attr:$seed);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$dtype, "float":$high, "float":$low, "float":$seed), [{

        $_state.addOperands(values);
        $_state.addAttribute("dtype",$_builder.getI64IntegerAttr(dtype));
        $_state.addAttribute("high",$_builder.getF32FloatAttr(high));
        $_state.addAttribute("low",$_builder.getF32FloatAttr(low));
        $_state.addAttribute("seed",$_builder.getF32FloatAttr(seed));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reciprocal : Poptorch_Op<"reciprocal", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reducel1 : Poptorch_Op<"reducel1", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];
}

def Poptorch_reducel2 : Poptorch_Op<"reducel2", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reducelogsum : Poptorch_Op<"reducelogsum", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reducelogsumexp : Poptorch_Op<"reducelogsumexp", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];
}

def Poptorch_reducemax : Poptorch_Op<"reducemax", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reducemin : Poptorch_Op<"reducemin", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reduceprod : Poptorch_Op<"reduceprod", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_reducesumsquare : Poptorch_Op<"reducesumsquare", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes, I64Attr:$keepdims);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes, "std::int64_t":$keepdims), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdims",$_builder.getI64IntegerAttr(keepdims));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}
def Poptorch_selu : Poptorch_Op<"selu", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, F32Attr:$alpha, F32Attr:$gamma);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "float":$alpha, "float":$gamma), [{

        $_state.addOperands(values);
        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));
        $_state.addAttribute("gamma",$_builder.getF32FloatAttr(gamma));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_shape : Poptorch_Op<"shape", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_size : Poptorch_Op<"size", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_softsign : Poptorch_Op<"softsign", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_spacetodepth : Poptorch_Op<"spacetodepth", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64Attr:$blocksize);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$blocksize), [{

        $_state.addOperands(values);
        $_state.addAttribute("blocksize",$_builder.getI64IntegerAttr(blocksize));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_squeeze : Poptorch_Op<"squeeze", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$axes);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$axes), [{

        $_state.addOperands(values);
        $_state.addAttribute("axes",$_builder.getI64ArrayAttr(axes));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_tile : Poptorch_Op<"tile", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values), [{

        $_state.addOperands(values);

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_ones : Poptorch_Op<"ones", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$shape, StrAttr:$dtype);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$shape, "const char*":$dtype), [{

        $_state.addOperands(values);
        $_state.addAttribute("shape",$_builder.getI64ArrayAttr(shape));
        $_state.addAttribute("dtype",$_builder.getStringAttr(dtype));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}

def Poptorch_zeros : Poptorch_Op<"zeros", []> {
  let arguments = (ins Variadic<Poptorch_tensor>:$inputs, I64ArrayAttr:$shape, StrAttr:$dtype);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "const std::vector<std::int64_t>&":$shape, "const char*":$dtype), [{
        $_state.addAttribute("shape",$_builder.getI64ArrayAttr(shape));
        $_state.addAttribute("dtype",$_builder.getStringAttr(dtype));

        // Just take the first value type until this op is manually added.

        $_state.addTypes(values[0].getType());

  }]>];

}
