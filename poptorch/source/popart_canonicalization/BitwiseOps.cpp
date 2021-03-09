// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {
torch::jit::Node *bitwiseNotHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  if (allInputsBool(node)) {
    return createLogical_not(graph, {node->input(0)});
  }
  if (allInputsInteger(node)) {
    return createBitwisenot(graph, {node->input(0)});
  }
  return nullptr;
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::bitwise_not, bitwiseNotHandler);
}
} // namespace poptorch
