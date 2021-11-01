// Copyright (c) 2020 Graphcore Ltd. All rights reserved
#include "PoptorchSymbols.hpp"
#include "PoptorchStaticInit.hpp"
#include "poptorch_logging/Logging.hpp"

#define SYMBOL_INIT(Namespace, Name)                                           \
  Name = c10::Symbol::fromQualString(#Namespace "::" #Name)

namespace c10::aten {

c10::Symbol floor_divide;                // NOLINT
c10::Symbol prelu_;                      // NOLINT
c10::Symbol leaky_relu_;                 // NOLINT
c10::Symbol elu_;                        // NOLINT
c10::Symbol selu_;                       // NOLINT
c10::Symbol isinf;                       // NOLINT
c10::Symbol uniform_;                    // NOLINT
c10::Symbol normal_;                     // NOLINT
c10::Symbol where_;                      // NOLINT
c10::Symbol poisson_nll_loss;            // NOLINT
c10::Symbol multilabel_soft_margin_loss; // NOLINT
c10::Symbol bernoulli_;                  // NOLINT
c10::Symbol clamp_min_;                  // NOLINT
c10::Symbol clamp_max_;                  // NOLINT
c10::Symbol one_hot;                     // NOLINT
c10::Symbol pow_;                        // NOLINT
c10::Symbol scatter_add_;                // NOLINT
c10::Symbol feature_dropout_;            // NOLINT
c10::Symbol roll;                        // NOLINT
c10::Symbol nll_loss_nd;                 // NOLINT
c10::Symbol cross_entropy_loss;          // NOLINT
c10::Symbol new_ones;                    // NOLINT
c10::Symbol new_zeros;                   // NOLINT
c10::Symbol numpy_T;                     // NOLINT
c10::Symbol _reshape_alias;              // NOLINT

// Removed in 1.10.
#if TORCH_MINOR_VERSION < 10
c10::Symbol amax;        // NOLINT
c10::Symbol amin;        // NOLINT
c10::Symbol logical_not; // NOLINT
c10::Symbol logical_and; // NOLINT
c10::Symbol logical_or;  // NOLINT
c10::Symbol logical_xor; // NOLINT
#endif

// clang-format off
__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializeAtenSymbols() {
  // clang-format on
  poptorch::logging::trace("Initializing aten symbols");
  SYMBOL_INIT(aten, floor_divide);
  SYMBOL_INIT(aten, prelu_);
  SYMBOL_INIT(aten, leaky_relu_);
  SYMBOL_INIT(aten, elu_);
  SYMBOL_INIT(aten, selu_);
  SYMBOL_INIT(aten, isinf);
  SYMBOL_INIT(aten, uniform_);
  SYMBOL_INIT(aten, normal_);
  SYMBOL_INIT(aten, where_);
  SYMBOL_INIT(aten, poisson_nll_loss);
  SYMBOL_INIT(aten, multilabel_soft_margin_loss);
  SYMBOL_INIT(aten, bernoulli_);
  SYMBOL_INIT(aten, clamp_min_);
  SYMBOL_INIT(aten, clamp_max_);
  SYMBOL_INIT(aten, one_hot);
  SYMBOL_INIT(aten, pow_);
  SYMBOL_INIT(aten, scatter_add_);
  SYMBOL_INIT(aten, feature_dropout_);
  SYMBOL_INIT(aten, roll);
  SYMBOL_INIT(aten, nll_loss_nd);
  SYMBOL_INIT(aten, cross_entropy_loss);
  SYMBOL_INIT(aten, new_ones);
  SYMBOL_INIT(aten, new_zeros);
  SYMBOL_INIT(aten, numpy_T);
  SYMBOL_INIT(aten, _reshape_alias);

#if TORCH_MINOR_VERSION < 10
  SYMBOL_INIT(aten, amax);
  SYMBOL_INIT(aten, amin);
  SYMBOL_INIT(aten, logical_not);
  SYMBOL_INIT(aten, logical_and);
  SYMBOL_INIT(aten, logical_or);
  SYMBOL_INIT(aten, logical_xor);
#endif
}

} // namespace c10::aten

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
c10::Symbol begin_autocast;
c10::Symbol suppress_autocast;
c10::Symbol restore_autocast;
c10::Symbol autocast;

c10::Symbol update_param_inplace;

c10::Symbol host_side_cast;

c10::Symbol end_if;
c10::Symbol start_if_true;
c10::Symbol start_if_false;
c10::Symbol start_for_loop;
c10::Symbol end_for_loop;
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

// clang-format off
__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializePoptorchSymbols() {
  // clang-format on
  logging::trace("Initializing poptorch symbols");
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
  SYMBOL_INIT(poptorch, begin_autocast);
  SYMBOL_INIT(poptorch, suppress_autocast);
  SYMBOL_INIT(poptorch, restore_autocast);
  SYMBOL_INIT(poptorch, host_side_cast);
  SYMBOL_INIT(poptorch, autocast);

  SYMBOL_INIT(poptorch, update_param_inplace);

  SYMBOL_INIT(poptorch, end_if);
  SYMBOL_INIT(poptorch, start_if_true);
  SYMBOL_INIT(poptorch, start_if_false);
  SYMBOL_INIT(poptorch, start_for_loop);
  SYMBOL_INIT(poptorch, end_for_loop);
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
}

} // namespace poptorch
} // namespace symbols

c10::Symbol getOverlapSymbol(const char *suffix, unsigned int num) {
  std::stringstream overlap_ss;
  overlap_ss << "poptorch_overlap" << suffix << num;
  return c10::Symbol::attr(overlap_ss.str());
}

} // namespace poptorch
