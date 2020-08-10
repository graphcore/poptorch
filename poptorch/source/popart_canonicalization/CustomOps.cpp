// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *customOpHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  std::vector<torch::jit::Value *> inputs =
      handleTensorList(node->input(0)->node());
  std::string name = constantToString(node->input(1)->node());
  std::string domain = constantToString(node->input(2)->node());

  // Get the domain version.
  std::int64_t domain_version = constantToLong(node->input(3)->node());

  // Get the number of outputs.
  std::int64_t num_outputs = constantToLong(node->input(4)->node());

  // Add the custom op with a variadic number of outputs.
  torch::jit::Node *custom_op = createCustomOperation(
      graph, inputs, name, domain, domain_version, num_outputs);

  // It is replacing an operation which returned a list so add a list
  // construct to keep the IR legal.
  return createAndInsertNode(graph, at::prim::ListConstruct,
                             custom_op->outputs());
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    symbols::poptorch::custom_operation, customOpHandler);
// clang-format on

} // namespace poptorch
