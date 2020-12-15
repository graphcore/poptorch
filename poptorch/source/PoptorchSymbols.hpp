// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef SOURCE_POPTORCH_SYMBOLS_H
#define SOURCE_POPTORCH_SYMBOLS_H
#include <torch/csrc/jit/ir/ir.h>

// Create all the C10 symbols.
// For some reason aten::relu_ is missing from the c10 namespace
namespace c10::aten {
extern c10::Symbol relu_;
extern c10::Symbol dropout_;
extern c10::Symbol hardtanh_;
extern c10::Symbol logical_not;
extern c10::Symbol floor_divide;
extern c10::Symbol true_divide;
extern c10::Symbol prelu_;
extern c10::Symbol leaky_relu_;
extern c10::Symbol clamp_;
extern c10::Symbol elu_;
extern c10::Symbol selu_;
extern c10::Symbol isnan;
extern c10::Symbol isinf;
extern c10::Symbol uniform_;
extern c10::Symbol normal_;
extern c10::Symbol masked_fill_;
extern c10::Symbol where_;
extern c10::Symbol poisson_nll_loss;
extern c10::Symbol multilabel_soft_margin_loss;
extern c10::Symbol bernoulli_;
} // namespace c10::aten

namespace poptorch {

namespace symbols {
#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs) \
  namespace Namespace {                                                  \
    extern c10::Symbol FuncName;                                         \
  }

#include "popart_compiler/SupportedOperations.inc.hpp"

#undef OP_DECL
} // namespace symbols

namespace symbols::poptorch {
extern c10::Symbol begin_ipu_block;
extern c10::Symbol end_ipu_block;
extern c10::Symbol identity_loss;
extern c10::Symbol set_available_memory;
extern c10::Symbol set_matmul_serialization;
extern c10::Symbol optimizer_group;
extern c10::Symbol begin_multi_conv;
extern c10::Symbol multi_conv_part;
extern c10::Symbol end_multi_conv;

// Casting is done before passing the input to the IPU: the op is used so that
// so that that input types match those received from pytorch but that the input
// types to later ops have the correct type.
extern c10::Symbol host_side_cast;

} // namespace symbols::poptorch

} // namespace poptorch

#endif // SOURCE_POPTORCH_SYMBOLS_H
