// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

void ipu_print_tensor::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "ipu_print_tensor"
                  " is currently unimplemented.");
}

void nop::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "nop"
                  " is currently unimplemented.");
}

void begin_ipu_block::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "begin_ipu_block"
                  " is currently unimplemented.");
}

void end_ipu_block::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "end_ipu_block"
                  " is currently unimplemented.");
}

void internal_cast::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "internal_cast"
                  " is currently unimplemented.");
}

void custom_operation::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "custom_operation"
                  " is currently unimplemented.");
}

void ctc_beam_search_decoder::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "ctc_beam_search_decoder"
                  " is currently unimplemented.");
}

void identity_loss::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "identity_loss"
                  " is currently unimplemented.");
}

void start_for_loop::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "start_for_loop"
                  " is currently unimplemented.");
}

void end_for_loop::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "end_for_loop"
                  " is currently unimplemented.");
}

void optimizer_group::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "optimizer_group"
                  " is currently unimplemented.");
}

void set_matmul_serialization::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "set_matmul_serialization"
                  " is currently unimplemented.");
}

void set_overlap_for_input::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "set_overlap_for_input"
                  " is currently unimplemented.");
}

void set_overlap_for_output::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "set_overlap_for_output"
                  " is currently unimplemented.");
}

void recomputation_checkpoint::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "recomputation_checkpoint"
                  " is currently unimplemented.");
}

void set_available_memory::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "set_available_memory"
                  " is currently unimplemented.");
}

void begin_multi_conv::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "begin_multi_conv"
                  " is currently unimplemented.");
}

void end_multi_conv::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "end_multi_conv"
                  " is currently unimplemented.");
}

void push_name_scope::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "push_name_scope"
                  " is currently unimplemented.");
}

void pop_name_scope::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "pop_name_scope"
                  " is currently unimplemented.");
}

void begin_autocast::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "begin_autocast"
                  " is currently unimplemented.");
}

void suppress_autocast::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "suppress_autocast"
                  " is currently unimplemented.");
}

void restore_autocast::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "restore_autocast"
                  " is currently unimplemented.");
}

void end_cpu_op::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "end_cpu_op"
                  " is currently unimplemented.");
}

void call_cpu_op::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "call_cpu_op"
                  " is currently unimplemented.");
}

void set_attribute::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "set_attribute"
                  " is currently unimplemented.");
}

void clear_attribute::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "clear_attribute"
                  " is currently unimplemented.");
}

} // namespace poptorch_ir
