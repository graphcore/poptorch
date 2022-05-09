// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

// PyTorch's bitwise_* functions can take any integral tensors as input (ie.
// torch.{uint8,int8,int16,int32,int64}. However, Poplibs' element-wise binary
// ops don't support 8-bit int inputs (see
// popops/codelets/elementwiseBinaryCodelets.cpp). Use this extra function to
// generate slightly nicer error messages.
void verifyCompatibleIntegralInputs(torch::jit::Node *node,
                                    const std::string &op_name) {
  ERROR_ON_MSG(allInputsOfType(node, at::ScalarType::Byte) ||
                   allInputsOfType(node, at::ScalarType::Char),
               op_name + ": Poplar does not support binary operations on "
                         "8-bit integral types.");
}

torch::jit::Node *bitwiseAndHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  if (allInputsBool(node)) {
    return createLogical_and(graph, {node->input(0), node->input(1)});
  }
  if (allInputsInteger(node)) {
    verifyCompatibleIntegralInputs(node, "Bitwise-and");
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
    verifyCompatibleIntegralInputs(node, "Bitwise-not");
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
    verifyCompatibleIntegralInputs(node, "Bitwise-or");
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
    verifyCompatibleIntegralInputs(node, "Bitwise-xor");
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
