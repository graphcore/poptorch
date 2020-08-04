// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *randHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool?
  // pin_memory) -> Tensor
  std::vector<int64_t> shape = shapeFromTensor(node->output(0));
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::random_uniform);
  new_node->is_(c10::attr::shape, shape);

  return new_node;
}
} // namespace

static bool handler = registerHandlers(c10::aten::rand, randHandler);
} // namespace poptorch
