// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *greaterLessEqualHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  torch::jit::Node *comparison = nullptr;
  torch::jit::Symbol kind = node->kind();
  torch::jit::Value *lhs = node->input(0);
  torch::jit::Value *rhs = node->input(1);

  // Node will either be < or >.
  if (kind == c10::aten::ge) {
    comparison = createGreater(graph, {lhs, rhs});
  } else {
    comparison = createLess(graph, {lhs, rhs});
  }

  // We do a check for ==
  torch::jit::Node *equal = createEqual(graph, {lhs, rhs});

  // The final node will be a combination of equals and less or greater.
  return createLogical_or(graph, {equal->output(), comparison->output()});
}

torch::jit::Node *notEqualHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  torch::jit::Value *lhs = node->input(0);
  torch::jit::Value *rhs = node->input(1);

  // Not(equal(lhs, rhs))
  torch::jit::Node *equal = createEqual(graph, {lhs, rhs});
  return createLogical_not(graph, {equal->output()});
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::ge, greaterLessEqualHandler);
  registerHandler(c10::aten::le, greaterLessEqualHandler);
  registerHandler(c10::aten::ne, notEqualHandler);
}

} // namespace poptorch
