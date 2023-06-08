// Copyright (c) 2020 Graphcore Ltd. All rights reserved
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "PoptorchStaticInit.hpp"
#include "PoptorchSymbols.hpp"
#include "poptorch_logging/Logging.hpp"

#define SYMBOL_INIT(Namespace, Name)                                           \
  Name = c10::Symbol::fromQualString(#Namespace "::" #Name)

namespace c10::aten {

c10::Symbol multilabel_soft_margin_loss; // NOLINT

// clang-format off
__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializeAtenSymbols() {
  // clang-format on
  poptorch::logging::trace("Initializing aten symbols");
  SYMBOL_INIT(aten, multilabel_soft_margin_loss);
}

} // namespace c10::aten

namespace torch_scatter {

c10::Symbol scatter_max; // NOLINT
c10::Symbol scatter_min; // NOLINT
c10::Symbol scatter_mul; // NOLINT

// clang-format off
__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializeTorchScatterSymbols() {
  // clang-format on
  poptorch::logging::trace("Initializing torch_scatter symbols");
  SYMBOL_INIT(torch_scatter, scatter_max);
  SYMBOL_INIT(torch_scatter, scatter_min);
  SYMBOL_INIT(torch_scatter, scatter_mul);
}

} // namespace torch_scatter

namespace torch_cluster {

c10::Symbol grid; // NOLINT

// clang-format off
__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializeTorchScatterSymbols() {
  // clang-format on
  poptorch::logging::trace("Initializing torch_scatter symbols");
  SYMBOL_INIT(torch_cluster, grid);
}

} // namespace torch_cluster

namespace torch_spline_conv {

c10::Symbol spline_basis;     // NOLINT
c10::Symbol spline_weighting; // NOLINT

// clang-format off
  __attribute__((constructor(SYMBOL_INIT_PRIORITY)))
  static void initializeTorchSplineConvSymbols() {
  // clang-format on
  poptorch::logging::trace("Initializing torch_spline_conv symbols");
  SYMBOL_INIT(torch_spline_conv, spline_basis);
  SYMBOL_INIT(torch_spline_conv, spline_weighting);
}

} // namespace torch_spline_conv

namespace poptorch {
namespace symbols {

#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs)       \
  c10::Symbol Namespace::FuncName;

#define OP_DECL_NO_RETURN(Namespace, FuncName, function, OnnxImpl, Args,       \
                          BodyArgs)                                            \
  c10::Symbol Namespace::FuncName;

#include "popart_compiler/SupportedOperations.inc.hpp" // NOLINT

#undef OP_DECL
#undef OP_DECL_NO_RETURN
// clang-format off
__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializeSupportedOperations() {
  // clang-format on
  logging::trace("Initializing supported operationss");

#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs)       \
  Namespace::FuncName =                                                        \
      c10::Symbol::fromQualString(#Namespace "::" #FuncName); // NOLINT

#define OP_DECL_NO_RETURN(Namespace, FuncName, function, OnnxImpl, Args,       \
                          BodyArgs)                                            \
  Namespace::FuncName =                                                        \
      c10::Symbol::fromQualString(#Namespace "::" #FuncName); // NOLINT

#include "popart_compiler/SupportedOperations.inc.hpp" // NOLINT

#undef OP_DECL
#undef OP_DECL_NO_RETURN
}

namespace poptorch {

c10::Symbol nop;
c10::Symbol dynamic_slice;
c10::Symbol dynamic_update;
c10::Symbol begin_ipu_block;
c10::Symbol internal_cast;
c10::Symbol end_ipu_block;
c10::Symbol identity_loss;
c10::Symbol set_available_memory;
c10::Symbol set_matmul_serialization;
c10::Symbol set_overlap_for_input;
c10::Symbol set_overlap_for_output;
c10::Symbol optimizer_group;
c10::Symbol begin_multi_conv;
c10::Symbol multi_conv_part;
c10::Symbol end_multi_conv;

c10::Symbol update_param_inplace;

c10::Symbol host_side_cast;

c10::Symbol start_for_loop;
c10::Symbol end_for_loop;

c10::Symbol start_if_block;
c10::Symbol start_else_block;
c10::Symbol end_if_block;

c10::Symbol push_name_scope;
c10::Symbol pop_name_scope;
c10::Symbol add_untyped_input_tensor;

c10::Symbol host_and_ipu_side_tensor_constant;

c10::Symbol call_cpu_op;
c10::Symbol end_cpu_op;

c10::Symbol canonicalised_cpu_call;
c10::Symbol ctc_beam_search_decoder;

c10::Symbol set_attribute;
c10::Symbol clear_attribute;

c10::Symbol fps;

// clang-format off
__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializePoptorchSymbols() {
  // clang-format on
  logging::trace("Initializing poptorch symbols");
  SYMBOL_INIT(poptorch, nop);
  SYMBOL_INIT(poptorch, dynamic_slice);
  SYMBOL_INIT(poptorch, dynamic_update);
  SYMBOL_INIT(poptorch, begin_ipu_block);
  SYMBOL_INIT(poptorch, internal_cast);
  SYMBOL_INIT(poptorch, end_ipu_block);
  SYMBOL_INIT(poptorch, identity_loss);
  SYMBOL_INIT(poptorch, set_available_memory);
  SYMBOL_INIT(poptorch, set_matmul_serialization);
  SYMBOL_INIT(poptorch, set_overlap_for_input);
  SYMBOL_INIT(poptorch, set_overlap_for_output);
  SYMBOL_INIT(poptorch, optimizer_group);
  SYMBOL_INIT(poptorch, begin_multi_conv);
  SYMBOL_INIT(poptorch, multi_conv_part);
  SYMBOL_INIT(poptorch, end_multi_conv);
  SYMBOL_INIT(poptorch, host_side_cast);

  SYMBOL_INIT(poptorch, update_param_inplace);

  SYMBOL_INIT(poptorch, start_for_loop);
  SYMBOL_INIT(poptorch, end_for_loop);

  SYMBOL_INIT(poptorch, start_if_block);
  SYMBOL_INIT(poptorch, start_else_block);
  SYMBOL_INIT(poptorch, end_if_block);

  SYMBOL_INIT(poptorch, push_name_scope);
  SYMBOL_INIT(poptorch, pop_name_scope);
  SYMBOL_INIT(poptorch, add_untyped_input_tensor);

  SYMBOL_INIT(poptorch, host_and_ipu_side_tensor_constant);

  SYMBOL_INIT(poptorch, call_cpu_op);
  SYMBOL_INIT(poptorch, end_cpu_op);

  SYMBOL_INIT(poptorch, canonicalised_cpu_call);
  SYMBOL_INIT(poptorch, ctc_beam_search_decoder);

  SYMBOL_INIT(poptorch, set_attribute);
  SYMBOL_INIT(poptorch, clear_attribute);

  SYMBOL_INIT(poptorch, fps);
}

} // namespace poptorch
} // namespace symbols

c10::Symbol getOverlapSymbol(const char *suffix, unsigned int num) {
  return c10::Symbol::attr(
      fmt::format("poptorch_overlap_for_{}{}", suffix, num));
}

} // namespace poptorch
