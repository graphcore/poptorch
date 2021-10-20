// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {

std::map<torch::jit::Value *, torch::jit::Value *> detached_values;
std::set<torch::jit::Node *> visited_nodes;

torch::jit::Value *possiblyDetachedValue(torch::jit::Graph *graph,
                                         torch::jit::Value *value) {
  auto *producer = value->node();
  auto producer_kind = producer->kind();

  if (value->requires_grad() || producer_kind == c10::prim::Constant ||
      producer_kind == symbols::poptorch::tensor_constant ||
      producer_kind == symbols::poptorch::host_side_tensor_constant ||
      producer_kind == symbols::popart::detach ||
      producer_kind == c10::prim::TupleConstruct ||
      producer_kind == c10::prim::ListConstruct) {
    return value;
  }

  auto it = detached_values.find(value);
  if (it == detached_values.end()) {
    auto *detach = graph->create(symbols::popart::detach);
    detach->addInput(value);
    detach->insertAfter(producer);
    it = detached_values.insert({value, detach->output(0)}).first;
  }

  return it->second;
}

void maybeInsertDetachOp(torch::jit::Graph *graph, torch::jit::Node *node) {
  logging::LogContext ctx(
      "AddDetachOperations (maybeInsertDetachOp) processing " +
      nodeToString(node));

  if (visited_nodes.find(node) != visited_nodes.end()) {
    return;
  }
  visited_nodes.insert(node);

  for (torch::jit::Value *input : node->inputs()) {
    auto *detach = possiblyDetachedValue(graph, input);
    if (input == detach) {
      maybeInsertDetachOp(graph, input->node());
    }
  }
}

void replaceDetachedValues(torch::jit::Node *node) {
  logging::LogContext ctx(
      "AddDetachOperations (replaceDetachedValues) processing " +
      nodeToString(node));

  if (visited_nodes.find(node) != visited_nodes.end()) {
    return;
  }
  visited_nodes.insert(node);

  for (torch::jit::Value *input : node->inputs()) {
    auto it = detached_values.find(input);
    if (it != detached_values.end()) {
      if (node->kind() == symbols::popart::detach) {
        // Only replace values (with their detached counterparts) that exist
        // after the detach node that generated the detached value.
        return;
      }
      node->replaceInputWith(input, it->second);
    }
    replaceDetachedValues(input->node());
  }
}

} // namespace

void addDetachOperations(torch::jit::Graph *graph) {
  detached_values.clear();
  visited_nodes.clear();

  // Special prim::Param nodes that correspond to graph inputs should not be
  // detached so we superficially mark them as detached before processing.
  for (torch::jit::Value *input : graph->inputs()) {
    visited_nodes.insert(input->node());
    detached_values.insert({input, input});
  }

  // Process the graph recursively and replace the values at the end.
  maybeInsertDetachOp(graph, graph->return_node());

  visited_nodes.clear();
  for (torch::jit::Value *input : graph->inputs()) {
    visited_nodes.insert(input->node());
  }
  replaceDetachedValues(graph->return_node());
}

} // namespace poptorch
