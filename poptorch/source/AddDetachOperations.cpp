// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

using UsesToReplace =
    std::vector<std::pair<torch::jit::Value *, torch::jit::Value *>>;

const c10::Symbol visited_node_attr =
    c10::Symbol::fromQualString("attr::visited_node");

void markNodeVisited(torch::jit::Node *node) {
  if (!node->hasAttribute(visited_node_attr)) {
    node->i_(visited_node_attr, 1);
  }
}

void unmarkNodeVisited(torch::jit::Node *node) {
  if (node->hasAttribute(visited_node_attr)) {
    node->removeAttribute(visited_node_attr);
  }
}

bool isMarkedVisited(torch::jit::Node *node) {
  return node->hasAttribute(visited_node_attr) &&
         node->i(visited_node_attr) == 1;
}

void maybeInsertDetachOp(torch::jit::Graph *graph, torch::jit::Value *value,
                         UsesToReplace *uses_to_replace) {
  auto producer = value->node();
  logging::LogContext ctx("AddDetachOperations processing " +
                          nodeToString(producer));

  auto producer_kind = producer->kind();

  if (isMarkedVisited(producer) || producer_kind == c10::prim::Constant ||
      producer_kind == symbols::poptorch::tensor_constant ||
      producer_kind == symbols::poptorch::host_side_tensor_constant) {
    return;
  }

  markNodeVisited(producer);

  if (!value->requires_grad() && !(producer_kind == c10::prim::TupleConstruct ||
                                   producer_kind == c10::prim::ListConstruct)) {
    // All node's outputs either requires_grad=True or don't, so we process all
    // of them now and skip processing the next time they are potentially
    // encountered.
    for (auto output : producer->outputs()) {
      auto detach = createDetach(graph, {output});
      detach->moveAfter(producer);
      uses_to_replace->emplace_back(output, detach->output(0));
    }
    return;
  }

  for (torch::jit::Value *input : producer->inputs()) {
    maybeInsertDetachOp(graph, input, uses_to_replace);
  }
}

void addDetachOperations(torch::jit::Graph *graph) {
  // Special prim::Param nodes that correspond to graph inputs should not be
  // visited so we superficially mark them as visited before processing.
  for (torch::jit::Value *input : graph->inputs()) {
    markNodeVisited(input->node());
  }

  // Process the graph recursively and replace the uses at the end.
  UsesToReplace uses_to_replace;
  for (torch::jit::Value *output : graph->outputs()) {
    maybeInsertDetachOp(graph, output, &uses_to_replace);
  }
  for (const auto &use : uses_to_replace) {
    use.first->replaceAllUsesAfterNodeWith(use.second->node(), use.second);
  }

  // Cleaning up.
  for (torch::jit::Node *node : graph->nodes()) {
    unmarkNodeVisited(node);
  }
}

} // namespace poptorch
