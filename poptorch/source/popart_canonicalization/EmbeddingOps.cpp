// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *embeddingHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool
  // scale_grad_by_freq, bool sparse) -> Tensor

  bool scale_grad_by_freq = constantToBool(node->input(3)->node());
  bool sparse = constantToBool(node->input(4)->node());

  ERROR_ON_MSG(scale_grad_by_freq || sparse,
               "Unsupported aten::embedding operation");

  return createGather(graph, {node->input(0), node->input(1)}, 0);
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::embedding, embeddingHandler);
}

} // namespace poptorch
