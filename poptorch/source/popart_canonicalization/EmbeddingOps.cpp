// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

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

torch::jit::Node *embeddingBagHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  // aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool
  // scale_grad_by_freq, int mode, bool sparse, Tensor per_sample_weights, bool
  // include_last_offset) -> Tensor

  bool scale_grad_by_freq = constantToBool(node->input(3)->node());
  bool sparse = constantToBool(node->input(5)->node());

  ERROR_ON_MSG(scale_grad_by_freq || sparse,
               "Unsupported aten::embedding_bag operation");

  // aten::embedding_bag has 4 outputs but only the first one is used so we
  // delete them here to match our output
  while (node->outputs().size() > 1) {
    node->eraseOutput(node->outputs().size() - 1);
  }

  auto weight = node->input(0);
  auto indices = node->input(1);
  auto offsets = node->input(2);
  int64_t mode = constantToLong(node->input(4)->node());
  auto per_sample_weights = node->input(6);
  bool include_last_offset = constantToBool(node->input(7)->node());

  auto reduction = [mode](torch::jit::Graph *g, torch::jit::Value *v) {
    if (mode == 0) {
      return createReducesum(g, {v}, {0}, 1)->output();
    }
    if (mode == 1) {
      return createReducemean(g, {v}, {0}, 1)->output();
    }
    return createReducemax(g, {v}, {0}, 1)->output();
  };

  ERROR_ON_MSG(!isTensorConstant(offsets->node()),
               "Unsupported aten::embedding_bag operation: offsets tensor must "
               "be a constant.");
  auto offsets_tensor = offsets->node()->t(c10::attr::value);

  if (!include_last_offset) {
    // Append INT_MAX to use as the last offset slice
    offsets_tensor = at::cat({offsets_tensor, at::tensor(INT_MAX)});
  }

  auto slices = offsets_tensor.accessor<int32_t, 1>();
  auto zero = wrapInConstant1D(graph, 0);
  torch::jit::value_list values;

  // Use the offsets to extract each bag from the indices.
  // For each bag: Gather then reduce from the embedding matrix
  for (int64_t i = 0; i < offsets_tensor.size(0) - 1; i++) {
    auto start = wrapInConstant1D(graph, static_cast<int64_t>(slices[i]));
    auto end = wrapInConstant1D(graph, static_cast<int64_t>(slices[i + 1]));
    auto bag = createSlice(graph, {indices, start, end, zero})->output();
    auto gather = createGather(graph, {weight, bag}, 0)->output();

    if (!isNone(per_sample_weights)) {
      auto psw =
          createSlice(graph, {per_sample_weights, start, end, zero})->output();
      psw = createUnsqueeze(graph, {psw}, {1})->output();
      gather = createMul(graph, {gather, psw})->output();
    }

    values.push_back(reduction(graph, gather));
  }

  return createConcat(graph, values, 0);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::embedding, embeddingHandler);
  registerHandler(c10::aten::embedding_bag, embeddingBagHandler);
}

} // namespace poptorch
