// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef SOURCE_POPTORCH_SYMBOLS_H
#define SOURCE_POPTORCH_SYMBOLS_H
#include <torch/csrc/jit/ir/ir.h>

// Create all the C10 Symbols.
#define SYMBOL_DECL(Namespace, FuncName)                                       \
  namespace Namespace {                                                        \
  const c10::Symbol FuncName(c10::Symbol::fromQualString(#Namespace            \
                                                         "::" #FuncName));     \
  }

// For some reason aten::relu_ is missing from the c10 namespace
namespace c10 {
SYMBOL_DECL(aten, relu_)
SYMBOL_DECL(aten, dropout_)
SYMBOL_DECL(aten, hardtanh_)
SYMBOL_DECL(aten, logical_not)
SYMBOL_DECL(aten, floor_divide)
SYMBOL_DECL(aten, true_divide)
SYMBOL_DECL(aten, leaky_relu_)
SYMBOL_DECL(aten, elu_)
SYMBOL_DECL(aten, selu_)
} // namespace c10

namespace poptorch {

namespace Symbols {
#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs)       \
  SYMBOL_DECL(Namespace, FuncName)

#include "popart_compiler/SupportedOperations.inc.h"
SYMBOL_DECL(poptorch, begin_ipu_block)
SYMBOL_DECL(poptorch, end_ipu_block)
SYMBOL_DECL(poptorch, identity_loss)
#undef OP_DECL
#undef SYMBOL_DECL
} // namespace Symbols

} // namespace poptorch

#endif // SOURCE_POPTORCH_SYMBOLS_H
