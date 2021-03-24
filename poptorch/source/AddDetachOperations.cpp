// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {

using UsesToReplace =
    std::vector<std::pair<torch::jit::Value *, torch::jit::Value *>>;

const c10::Symbol detached_node_attr =
    c10::Symbol::fromQualString("attr::detached_node");

void markNodeDetached(torch::jit::Node *node) {
  if (!node->hasAttribute(detached_node_attr)) {
    node->i_(detached_node_attr, 1);
  }
}

void unmarkNodeDetached(torch::jit::Node *node) {
  if (node->hasAttribute(detached_node_attr)) {
    node->removeAttribute(detached_node_attr);
  }
}

bool isMarkedDetached(torch::jit::Node *node) {
  return node->hasAttribute(detached_node_attr) &&
         node->i(detached_node_attr) == 1;
}

void maybeInsertDetachOp(torch::jit::Graph *graph, torch::jit::Value *value,
                         UsesToReplace *uses_to_replace) {
  auto producer = value->node();
  logging::LogContext ctx("AddDetachOperations processing " +
                          nodeToString(producer));

  auto producer_kind = producer->kind();

  if (producer_kind == c10::prim::Constant ||
      producer_kind == symbols::poptorch::tensor_constant ||
      producer_kind == symbols::poptorch::host_side_tensor_constant) {
    return;
  }

  if (!isMarkedDetached(producer) && !value->requires_grad() &&
      !(producer_kind == c10::prim::TupleConstruct ||
        producer_kind == c10::prim::ListConstruct)) {
    // All node's outputs either requires_grad=True or don't, so we process all
    // of them now and skip processing the next time they are potentially
    // encountered.
    for (auto output : producer->outputs()) {
      auto detach = createDetach(graph, {output});
      detach->moveAfter(producer);
      uses_to_replace->emplace_back(output, detach->output(0));
    }
    markNodeDetached(producer);
    return;
  }

  for (torch::jit::Value *input : producer->inputs()) {
    maybeInsertDetachOp(graph, input, uses_to_replace);
  }
}

void addDetachOperations(torch::jit::Graph *graph) {
  // Special prim::Param nodes that correspond to graph inputs should not be
  // detached so we superficially mark them as detached before processing.
  for (torch::jit::Value *input : graph->inputs()) {
    markNodeDetached(input->node());
  }

  // Process the graph recursively and replace the uses at the end.
  UsesToReplace uses_to_replace;
  for (torch::jit::Value *output : graph->outputs()) {
    maybeInsertDetachOp(graph, output, &uses_to_replace);
  }
  for (const auto &use : uses_to_replace) {
    use.first->replaceAllUsesAfterNodeWith(use.second->node(), use.second);
    unmarkNodeDetached(use.first->node());
  }

  // Cleaning up.
  for (torch::jit::Value *input : graph->inputs()) {
    unmarkNodeDetached(input->node());
  }
}

} // namespace poptorch
