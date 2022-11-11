// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include <ATen/ATen.h>

namespace poptorch {
namespace {

torch::jit::Node *embeddingHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool
  // scale_grad_by_freq, bool sparse) -> Tensor

  const bool scale_grad_by_freq = constantToBool(node->input(3)->node());
  const bool sparse = constantToBool(node->input(4)->node());

  ERROR_ON_MSG(scale_grad_by_freq || sparse,
               "Unsupported aten::embedding operation");

  auto *weight = node->input(0);
  auto *indices = node->input(1);
  auto padding_idx = constantToLong(node->input(2)->node());

  if (padding_idx < 0) {
    // Default: padding_idx == -1 indicates no padding.
    return createGather(graph, {node->input(0), node->input(1)}, 0);
  }

  std::stringstream ss;
  ss << "{\"padding_idx\":" << std::to_string(padding_idx) << "}";

  auto *out = createCustomOperation(graph, {weight, indices}, "Embedding",
                                    "poptorch.custom_ops", 1, 1, ss.str());

  auto input_type = getNodeScalarType(weight);
  auto out_type = c10::TensorType::create(input_type, c10::nullopt,
                                          c10::nullopt, c10::nullopt);
  out->output(0)->setType(out_type);
  return out;
}

torch::jit::Node *embeddingBagHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  // aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool
  // scale_grad_by_freq, int mode, bool sparse, Tensor per_sample_weights, bool
  // include_last_offset, int? padding_idx) -> Tensor

  const bool scale_grad_by_freq = constantToBool(node->input(3)->node());
  const bool sparse = constantToBool(node->input(5)->node());
  auto *padding_idx = node->input(8);

  ERROR_ON_MSG(scale_grad_by_freq || sparse,
               "Unsupported aten::embedding_bag operation");

  if (!isNone(padding_idx)) {
    auto padding_idx_val = constantToInt(node->input(8)->node());
    ERROR_ON_MSG(padding_idx_val >= 0,
                 "Unsupported aten::embedding_bag operation: padding_idx "
                 "parameter is unsupported.");
  }

  // aten::embedding_bag has 4 outputs but only the first one is used so we
  // delete them here to match our output
  while (node->outputs().size() > 1) {
    node->eraseOutput(node->outputs().size() - 1);
  }

  auto *weight = node->input(0);
  auto *indices = node->input(1);
  auto *offsets = node->input(2);
  const int64_t mode = constantToLong(node->input(4)->node());
  auto *per_sample_weights = node->input(6);
  const bool include_last_offset = constantToBool(node->input(7)->node());

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
  auto offsets_tensor = getNodeTensorAttrValue(offsets->node());

  if (!include_last_offset) {
    // Append INT_MAX to use as the last offset slice
    offsets_tensor = at::cat({offsets_tensor, at::tensor(INT_MAX)});
  }

  auto slices = offsets_tensor.accessor<int32_t, 1>();
  torch::jit::value_list values;

  // Use the offsets to extract each bag from the indices.
  // For each bag: Gather then reduce from the embedding matrix
  for (int64_t i = 0; i < offsets_tensor.size(0) - 1; i++) {
    auto *bag = createSlice(graph, {indices}, {slices[i + 1]}, {slices[i]}, {0})
                    ->output();
    auto *gather = createGather(graph, {weight, bag}, 0)->output();

    if (!isNone(per_sample_weights)) {
      auto *psw = createSlice(graph, {per_sample_weights}, {slices[i + 1]},
                              {slices[i]}, {0})
                      ->output();
      psw = createUnsqueeze(graph, {psw}, {1})->output();
      gather = createMul(graph, {gather, psw})->output();
    }

    values.push_back(reduction(graph, gather));
  }

  return createConcat(graph, values, 0);
}

torch::jit::Node *onehotHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  torch::jit::Value *tensor = node->input(0);

  std::int64_t const num_classes = constantToLong(node->input(1)->node());

  ERROR_ON_MSG(num_classes == -1,
               "OneHot num classes must be specified and must be constant.");

  // The "hot/cold" values for the one hot representation.
  torch::jit::Node *values = createConstantInt(graph, {0, 1}, {2});

  torch::jit::Node *depth = createConstantInt(graph, {num_classes}, {});

  return createOnehot(graph, {tensor, depth->output(), values->output()}, -1);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::embedding, embeddingHandler);
  registerHandler(c10::aten::embedding_bag, embeddingBagHandler);
  registerHandler(c10::aten::_embedding_bag, embeddingBagHandler);
  registerHandler(c10::aten::one_hot, onehotHandler);
}

} // namespace poptorch
