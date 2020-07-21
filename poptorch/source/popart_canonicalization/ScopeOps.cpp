
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *beginIpuBlockHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  torch::jit::Node *new_node;
  // This could maybe be improved. Can we add attributes on the frontend?
  // TODO(T24134)
  new_node = createAndInsertNode(
      graph, c10::Symbol::fromQualString("poptorch::begin_ipu_block"), {},
      node->outputs().size());

  // Convert the prim::Constant into an attribute.
  std::int64_t ipu_id = *handleConstant<std::int64_t>(node->input()->node());
  new_node->i_(c10::Symbol::fromQualString("attr::ipu"), ipu_id);
  return new_node;
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    symbols::poptorch::begin_ipu_block, beginIpuBlockHandler);
// clang-format on

} // namespace poptorch
