// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {
torch::jit::Node *bitwiseAndHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  if (allInputsBool(node)) {
    return createLogical_and(graph, {node->input(0), node->input(1)});
  }
  if (allInputsInteger(node)) {
    return createBitwiseand(graph, {node->input(0), node->input(1)});
  }
  ERROR("Bitwise-and operator supports only bool and integer types");
  return nullptr;
}

torch::jit::Node *bitwiseNotHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  if (allInputsBool(node)) {
    return createLogical_not(graph, {node->input(0)});
  }
  if (allInputsInteger(node)) {
    return createBitwisenot(graph, {node->input(0)});
  }
  ERROR("Bitwise-not operator supports only bool and integer types");
  return nullptr;
}

torch::jit::Node *bitwiseOrHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  if (allInputsBool(node)) {
    return createLogical_or(graph, {node->input(0), node->input(1)});
  }
  if (allInputsInteger(node)) {
    return createBitwiseor(graph, {node->input(0), node->input(1)});
  }
  ERROR("Bitwise-or operator supports only bool and integer types");
  return nullptr;
}

torch::jit::Node *bitwiseXorHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  if (allInputsBool(node)) {
    return createLogical_xor(graph, {node->input(0), node->input(1)});
  }
  if (allInputsInteger(node)) {
    return createBitwisexor(graph, {node->input(0), node->input(1)});
  }
  ERROR("Bitwise-xor operator supports only bool and integer types");
  return nullptr;
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::bitwise_and, bitwiseAndHandler);
  registerHandler(c10::aten::bitwise_not, bitwiseNotHandler);
  registerHandler(c10::aten::bitwise_or, bitwiseOrHandler);
  registerHandler(c10::aten::bitwise_xor, bitwiseXorHandler);
  registerHandler(c10::aten::__and__, bitwiseAndHandler);
  registerHandler(c10::aten::__or__, bitwiseOrHandler);
  registerHandler(c10::aten::__xor__, bitwiseXorHandler);
}
} // namespace poptorch
