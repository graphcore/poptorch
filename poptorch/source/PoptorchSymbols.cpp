// Copyright (c) 2020 Graphcore Ltd. All rights reserved
#include "poptorch_logging/Logging.hpp"
#include "PoptorchStaticInit.hpp"
#include "PoptorchSymbols.hpp"

#define SYMBOL_INIT(Namespace, Name) \
    Name = c10::Symbol::fromQualString(#Namespace "::" #Name);

namespace c10::aten {

c10::Symbol relu_;
c10::Symbol dropout_;
c10::Symbol hardtanh_;
c10::Symbol logical_not;
c10::Symbol floor_divide;
c10::Symbol prelu_;
c10::Symbol leaky_relu_;
c10::Symbol elu_;
c10::Symbol selu_;
c10::Symbol isnan;
c10::Symbol isinf;
c10::Symbol uniform_;
c10::Symbol normal_;
c10::Symbol where_;
c10::Symbol poisson_nll_loss;
c10::Symbol multilabel_soft_margin_loss;
c10::Symbol bernoulli_;

__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializeAtenSymbols()
{
  logging::trace("Initializing aten symbols");
  SYMBOL_INIT(aten, relu_)
  SYMBOL_INIT(aten, dropout_)
  SYMBOL_INIT(aten, hardtanh_)
  SYMBOL_INIT(aten, logical_not)
  SYMBOL_INIT(aten, floor_divide)
  SYMBOL_INIT(aten, prelu_)
  SYMBOL_INIT(aten, leaky_relu_)
  SYMBOL_INIT(aten, elu_)
  SYMBOL_INIT(aten, selu_)
  SYMBOL_INIT(aten, isnan)
  SYMBOL_INIT(aten, isinf)
  SYMBOL_INIT(aten, uniform_)
  SYMBOL_INIT(aten, normal_)
  SYMBOL_INIT(aten, where_)
  SYMBOL_INIT(aten, poisson_nll_loss)
  SYMBOL_INIT(aten, multilabel_soft_margin_loss)
  SYMBOL_INIT(aten, bernoulli_)
}

} // namespace c10::aten

namespace poptorch::symbols {

#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs) \
    c10::Symbol Namespace::FuncName;

#include "popart_compiler/SupportedOperations.inc.hpp"

#undef OP_DECL

__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializeSupportedOperations()
{
    logging::trace("Initializing supported operationss");

#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs) \
    Namespace::FuncName = c10::Symbol::fromQualString(#Namespace "::" #FuncName);

#include "popart_compiler/SupportedOperations.inc.hpp"

#undef OP_DECL
}

}  // namespace poptorch::symbols

namespace poptorch::symbols::poptorch {

c10::Symbol begin_ipu_block;
c10::Symbol end_ipu_block;
c10::Symbol identity_loss;
c10::Symbol set_available_memory;
c10::Symbol set_matmul_serialization;
c10::Symbol optimizer_group;
c10::Symbol begin_multi_conv;
c10::Symbol multi_conv_part;
c10::Symbol end_multi_conv;
c10::Symbol host_side_cast;

__attribute__((constructor(SYMBOL_INIT_PRIORITY)))
static void initializePoptorchSymbols()
{
  logging::trace("Initializing poptorch symbols");
  SYMBOL_INIT(poptorch, begin_ipu_block)
  SYMBOL_INIT(poptorch, end_ipu_block)
  SYMBOL_INIT(poptorch, identity_loss)
  SYMBOL_INIT(poptorch, set_available_memory)
  SYMBOL_INIT(poptorch, set_matmul_serialization)
  SYMBOL_INIT(poptorch, optimizer_group)
  SYMBOL_INIT(poptorch, begin_multi_conv)
  SYMBOL_INIT(poptorch, multi_conv_part)
  SYMBOL_INIT(poptorch, end_multi_conv)
  SYMBOL_INIT(poptorch, host_side_cast)
}

} // namespace poptorch::symbols::poptorch

