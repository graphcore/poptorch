// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// ipu_print_tensor(Tensor self, str? title) -> Tensor
def Poptorch_ipu_print_tensor : Poptorch_NotImplementedOp<"ipu_print_tensor", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       OptionalAttr<StrAttr>:$title);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "std::optional<const char *>":$title), [{
          $_state.addOperands({self});
          if (title.has_value()) {
            $_state.addAttribute("title", $_builder.getStringAttr(title.value()));
          }
          $_state.addTypes({self.getType()});
     }]>
    ];
}

// nop(Tensor self) -> Tensor
def Poptorch_nop : Poptorch_NotImplementedOp<"nop", []> {
  let arguments = (ins Poptorch_tensor:$self);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self), [{
          $_state.addOperands({self});
          $_state.addTypes({self.getType()});
     }]>
    ];
}

// dynamic_slice(Tensor self, int dim, Tensor start, int size, int step) -> Tensor
def Poptorch_dynamic_slice : Poptorch_NotImplementedOp<"dynamic_slice", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$dim,
                       Poptorch_tensor:$start,
                       I64Attr:$size,
                       I64Attr:$step);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "int64_t":$dim,
                                 "mlir::Value":$start,
                                 "int64_t":$size,
                                 "int64_t":$step), [{
          $_state.addOperands({self, start});
          $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
          $_state.addAttribute("size", $_builder.getI64IntegerAttr(size));
          $_state.addAttribute("step", $_builder.getI64IntegerAttr(step));

          auto et = getElementType(self);
          auto shape = getShape(self);
          shape[dim] = (size + (step - 1)) / step;

          $_state.addTypes(mlir::RankedTensorType::get(shape, et));
     }]>
    ];
}

// begin_ipu_block(int stage_id, int phase_id, int ipu_id) -> ()
def Poptorch_begin_ipu_block : Poptorch_NotImplementedOp<"begin_ipu_block", []> {
  let arguments = (ins I64Attr:$stage_id,
                       I64Attr:$phase_id,
                       I64Attr:$ipu_id);
  let builders = [OpBuilder<(ins "int64_t":$stage_id,
                                    "int64_t":$phase_id,
                                    "int64_t":$ipu_id), [{
          $_state.addAttribute("stage_id", $_builder.getI64IntegerAttr(stage_id));
          $_state.addAttribute("phase_id", $_builder.getI64IntegerAttr(phase_id));
          $_state.addAttribute("ipu_id", $_builder.getI64IntegerAttr(ipu_id));
     }]>
    ];
}

// end_ipu_block() -> ()
def Poptorch_end_ipu_block : Poptorch_NotImplementedOp<"end_ipu_block", []> {
}

// internal_cast(Tensor self, str dtype) -> Tensor
def Poptorch_internal_cast : Poptorch_NotImplementedOp<"internal_cast", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       StrAttr:$dtype);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "const char *":$dtype), [{
          $_state.addOperands({self});
          $_state.addAttribute("dtype", $_builder.getStringAttr(dtype));
          $_state.addTypes({self.getType()});
     }]>
    ];
}

// custom_operation(Tensor[] inputs, str name, str domain, int domain_version, int num_outputs, Tensor(a!)[] outputs, str attributes) -> Tensor(a!)[]
def Poptorch_custom_operation : Poptorch_NotImplementedOp<"custom_operation", [AttrSizedOperandSegments]> {
  let arguments = (ins Poptorch_tensorlist:$inputs,
                       StrAttr:$name,
                       StrAttr:$domain,
                       I64Attr:$domain_version,
                       I64Attr:$num_outputs,
                       Poptorch_tensorlist:$outputs,
                       StrAttr:$attributes);
  let results = (outs Poptorch_tensorlist:$result);
  let builders = [OpBuilder<(ins "mlir::ValueRange":$inputs,
                                    "const char *":$name,
                                    "const char *":$domain,
                                    "int64_t":$domain_version,
                                    "int64_t":$num_outputs,
                                    "mlir::ValueRange":$outputs,
                                    "const char *":$attributes), [{
          std::vector<mlir::Value> operands(std::begin(inputs), std::end(inputs));
          operands.insert(std::end(operands), std::begin(outputs), std::end(outputs));
          $_state.addOperands(operands);
          $_state.addAttribute("name", $_builder.getStringAttr(name));
          $_state.addAttribute("domain", $_builder.getStringAttr(domain));
          $_state.addAttribute("domain_version", $_builder.getI64IntegerAttr(domain_version));
          $_state.addAttribute("num_outputs", $_builder.getI64IntegerAttr(num_outputs));
          $_state.addAttribute("attributes", $_builder.getStringAttr(attributes));

          std::vector<Type> types;
          for (const auto &o: outputs) {
            types.push_back(o.getType());
          }
          $_state.addTypes(types);
     }]>
    ];
}

// ctc_beam_search_decoder(Tensor probs, Tensor lengths, int blank, int beam_width, int top_paths) -> Tensor[]
def Poptorch_ctc_beam_search_decoder : Poptorch_NotImplementedOp<"ctc_beam_search_decoder", []> {
  let arguments = (ins Poptorch_tensor:$probs,
                       Poptorch_tensor:$lengths,
                       I64Attr:$blank,
                       I64Attr:$beam_width,
                       I64Attr:$top_paths);
  let results = (outs Poptorch_tensor:$outProbs, Poptorch_tensor:$outLengths, Poptorch_tensor:$outPaths);
  let builders = [OpBuilder<(ins "mlir::Value":$probs,
                                    "mlir::Value":$lengths,
                                    "int64_t":$blank,
                                    "int64_t":$beam_width,
                                    "int64_t":$top_paths), [{
          $_state.addOperands({probs, lengths});
          $_state.addAttribute("blank", $_builder.getI64IntegerAttr(blank));
          $_state.addAttribute("beam_width", $_builder.getI64IntegerAttr(beam_width));
          $_state.addAttribute("top_paths", $_builder.getI64IntegerAttr(top_paths));

          const auto &probsShape = getShape(probs);
          ERROR_ON_MSG(probsShape.size() != 3,
                      "Input probablities tensor must be rank-3 for "
                      "`ctc_beam_search_decoder`.");

          const auto input_size = probsShape[0];
          const auto batch_size = probsShape[1];

          auto outProbs = mlir::RankedTensorType::get({batch_size, top_paths},
                                                      getElementType(probs));
          auto outLengths = mlir::RankedTensorType::get({batch_size, top_paths},
                                                        getElementType(probs));
          auto outPaths =
            mlir::RankedTensorType::get({batch_size, top_paths, input_size},
                                        getElementType(probs));

          $_state.addTypes({outProbs, outLengths, outPaths});
     }]>
    ];
}

// identity_loss(Tensor x, str reduction) -> Tensor
def Poptorch_identity_loss : Poptorch_NotImplementedOp<"identity_loss", []> {
  let arguments = (ins Poptorch_tensor:$x,
                       I64Attr:$reduction);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$x,
                                    "int64_t":$reduction), [{
      $_state.addOperands({x});
      $_state.addAttribute("reduction", $_builder.getI64IntegerAttr(reduction));

      constexpr int64_t sum = 0;
      constexpr int64_t mean = 1;
      constexpr int64_t none = 2;

      switch (reduction) {
      case sum:
      case mean:
        $_state.addTypes({mlir::RankedTensorType::get({}, getElementType(x))});
        return;
      case none:
        $_state.addTypes({x.getType()});
        return;
      default:
        ERROR("reduction must be sum (0), mean (1) or none (2)");
      }
     }]>
    ];
}

// start_for_loop(Tensor[] inputs) -> ()
def Poptorch_start_for_loop : Poptorch_NotImplementedOp<"start_for_loop", []> {
  let arguments = (ins Poptorch_tensorlist:$inputs);
}

// end_for_loop(Tensor[] outputs, Tensor[] inputs, int trip_count) -> Tensor[]
def Poptorch_end_for_loop : Poptorch_NotImplementedOp<"end_for_loop", [AttrSizedOperandSegments]> {
  let arguments = (ins Poptorch_tensorlist:$outputs,
                       Poptorch_tensorlist:$inputs,
                       I64Attr:$trip_count);
  let results = (outs Poptorch_tensorlist:$result);
  let builders = [OpBuilder<(ins "mlir::ValueRange":$outputs,
                                 "mlir::ValueRange":$inputs,
                                 "int64_t":$trip_count), [{
          std::vector<mlir::Value> operands(std::begin(inputs), std::end(inputs));
          operands.insert(std::end(operands), std::begin(outputs), std::end(outputs));
          $_state.addOperands(operands);
          $_state.addAttribute("trip_count", $_builder.getI64IntegerAttr(trip_count));
          std::vector<Type> types;
          for (const auto &o: outputs) {
            types.push_back(o.getType());
          }
          $_state.addTypes(types);
     }]>
    ];
}

// optimizer_group(int group, Tensor[] inputs) -> ()
def Poptorch_optimizer_group : Poptorch_NotImplementedOp<"optimizer_group", []> {
  let arguments = (ins I64Attr:$group,
                       Poptorch_tensorlist:$inputs);
  let builders = [OpBuilder<(ins "int64_t":$group,
                                    "mlir::ValueRange":$inputs), [{
          $_state.addOperands({inputs});
          $_state.addAttribute("group", $_builder.getI64IntegerAttr(group));
     }]>
    ];
}

// set_matmul_serialization(Tensor matmul, str mode, int factor, bool keep_precision) -> Tensor
def Poptorch_set_matmul_serialization : Poptorch_NotImplementedOp<"set_matmul_serialization", []> {
  let arguments = (ins Poptorch_tensor:$matmul,
                       StrAttr:$mode,
                       I64Attr:$factor,
                       BoolAttr:$keep_precision);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$matmul,
                                    "const char *":$mode,
                                    "int64_t":$factor,
                                    "bool":$keep_precision), [{
          $_state.addOperands({matmul});
          $_state.addAttribute("mode", $_builder.getStringAttr(mode));
          $_state.addAttribute("factor", $_builder.getI64IntegerAttr(factor));
          $_state.addAttribute("keep_precision", $_builder.getBoolAttr(keep_precision));
          $_state.addTypes({matmul.getType()});
     }]>
    ];
}

// set_overlap_for_input(Tensor t, str mode) -> Tensor
def Poptorch_set_overlap_for_input : Poptorch_NotImplementedOp<"set_overlap_for_input", []> {
  let arguments = (ins Poptorch_tensor:$t,
                       StrAttr:$mode);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$t,
                                    "const char *":$mode), [{
          $_state.addOperands({t});
          $_state.addAttribute("mode", $_builder.getStringAttr(mode));
          $_state.addTypes({t.getType()});
     }]>
    ];
}

// set_overlap_for_output(Tensor t, str mode) -> Tensor
def Poptorch_set_overlap_for_output : Poptorch_NotImplementedOp<"set_overlap_for_output", []> {
  let arguments = (ins Poptorch_tensor:$t,
                       StrAttr:$mode);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$t,
                                    "const char *":$mode), [{
          $_state.addOperands({t});
          $_state.addAttribute("mode", $_builder.getStringAttr(mode));
          $_state.addTypes({t.getType()});
     }]>
    ];
}

// recomputation_checkpoint(Tensor self) -> Tensor
def Poptorch_recomputation_checkpoint : Poptorch_NotImplementedOp<"recomputation_checkpoint", []> {
  let arguments = (ins Poptorch_tensor:$self);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self), [{
          $_state.addOperands({self});
          $_state.addTypes({self.getType()});
     }]>
    ];
}

// set_available_memory(Tensor t, float mem) -> Tensor
def Poptorch_set_available_memory : Poptorch_NotImplementedOp<"set_available_memory", []> {
  let arguments = (ins Poptorch_tensor:$t,
                       F32Attr:$mem);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$t,
                                    "float":$mem), [{
          $_state.addOperands({t});
          $_state.addAttribute("mem", $_builder.getF32FloatAttr(mem));
          $_state.addTypes({t.getType()});
     }]>
    ];
}

// begin_multi_conv() -> ()
def Poptorch_begin_multi_conv : Poptorch_NotImplementedOp<"begin_multi_conv", []> {
}

// end_multi_conv(float[]? available_memory_proportions, int[]? partials_types, int? plan_type, int? per_conv_reserved_tiles, float? cycle_back_off, int[]? enableConvDithering) -> ()
def Poptorch_end_multi_conv : Poptorch_NotImplementedOp<"end_multi_conv", []> {
  let arguments = (ins OptionalAttr<F32ArrayAttr>:$available_memory_proportions,
                       OptionalAttr<I64ArrayAttr>:$partials_types,
                       OptionalAttr<I64Attr>:$plan_type,
                       OptionalAttr<I64Attr>:$per_conv_reserved_tiles,
                       OptionalAttr<F32Attr>:$cycle_back_off,
                       OptionalAttr<I64ArrayAttr>:$enableConvDithering);
  let builders = [OpBuilder<(ins "std::optional<std::vector<float>>":$available_memory_proportions,
                                    "std::optional<std::vector<int64_t>>":$partials_types,
                                    "std::optional<int64_t>":$plan_type,
                                    "std::optional<int64_t>":$per_conv_reserved_tiles,
                                    "std::optional<float>":$cycle_back_off,
                                    "std::optional<std::vector<int64_t>>":$enableConvDithering), [{
          if (available_memory_proportions.has_value()) {
            $_state.addAttribute("available_memory_proportions", $_builder.getF32ArrayAttr(available_memory_proportions.value()));
          }
          if (partials_types.has_value()) {
            $_state.addAttribute("partials_types", $_builder.getI64ArrayAttr(partials_types.value()));
          }
          if (plan_type.has_value()) {
            $_state.addAttribute("plan_type", $_builder.getI64IntegerAttr(plan_type.value()));
          }
          if (per_conv_reserved_tiles.has_value()) {
            $_state.addAttribute("per_conv_reserved_tiles", $_builder.getI64IntegerAttr(per_conv_reserved_tiles.value()));
          }
          if (cycle_back_off.has_value()) {
            $_state.addAttribute("cycle_back_off", $_builder.getF32FloatAttr(cycle_back_off.value()));
          }
          if (enableConvDithering.has_value()) {
            $_state.addAttribute("enableConvDithering", $_builder.getI64ArrayAttr(enableConvDithering.value()));
          }
     }]>
    ];
}

// push_name_scope(str name) -> ()
def Poptorch_push_name_scope : Poptorch_NotImplementedOp<"push_name_scope", []> {
  let arguments = (ins StrAttr:$name);
}

// pop_name_scope() -> ()
def Poptorch_pop_name_scope : Poptorch_NotImplementedOp<"pop_name_scope", []> {
}

// begin_autocast() -> ()
def Poptorch_begin_autocast : Poptorch_NotImplementedOp<"begin_autocast", []> {
}

// suppress_autocast() -> ()
def Poptorch_suppress_autocast : Poptorch_NotImplementedOp<"suppress_autocast", []> {
}

// restore_autocast() -> ()
def Poptorch_restore_autocast : Poptorch_NotImplementedOp<"restore_autocast", []> {
}

// end_cpu_op(Tensor[] output) -> Tensor[]
def Poptorch_end_cpu_op : Poptorch_NotImplementedOp<"end_cpu_op", []> {
  let arguments = (ins Poptorch_tensorlist:$output);
  let results = (outs Poptorch_tensorlist:$result);
  let builders = [OpBuilder<(ins "mlir::ValueRange":$output), [{
          $_state.addOperands({output});
          std::vector<Type> types;
          for (const auto &o: output) {
            types.push_back(o.getType());
          }
          $_state.addTypes(types);
     }]>
    ];
}

// call_cpu_op(Tensor[] inputs, str name) -> ()
def Poptorch_call_cpu_op : Poptorch_NotImplementedOp<"call_cpu_op", []> {
  let arguments = (ins Poptorch_tensorlist:$inputs,
                       StrAttr:$name);
}

// set_attribute(str attribute, str key, str value) -> ()
def Poptorch_set_attribute : Poptorch_NotImplementedOp<"set_attribute", []> {
  let arguments = (ins StrAttr:$attribute,
                       StrAttr:$key,
                       StrAttr:$value);
}

// clear_attribute(str attribute, str key) -> ()
def Poptorch_clear_attribute : Poptorch_NotImplementedOp<"clear_attribute", []> {
  let arguments = (ins StrAttr:$attribute,
                       StrAttr:$key);
}
