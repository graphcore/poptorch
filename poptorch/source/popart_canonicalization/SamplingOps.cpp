// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <random>

#include <ATen/ExpandUtils.h>

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *prepareMinimum(torch::jit::Graph *graph,
                                 torch::jit::Value *const lhs,
                                 torch::jit::Value *const rhs) {
  return createMin(graph, {lhs, rhs});
}

torch::jit::Node *prepareRowUpdate(torch::jit::Graph *graph,
                                   torch::jit::Value *const value,
                                   torch::jit::Value *const new_row,
                                   torch::jit::Value *const row_num,
                                   std::int64_t row_len) {
  return createDynamicupdate(graph, {value, row_num, new_row}, {0}, {row_len},
                             0);
}

torch::jit::Node *updateIdxs(torch::jit::Graph *graph,
                             torch::jit::Value *const idxs, std::int64_t offset,
                             torch::jit::Value *const new_val) {
  auto *const offset_node = createConstantLong(graph, {offset}, {1});
  return createDynamicupdate(graph, {idxs, offset_node->output(), new_val}, {0},
                             {1}, 0);
}

torch::jit::Node *prepareRowSlice(torch::jit::Graph *graph,
                                  torch::jit::Value *const value,
                                  torch::jit::Value *const row_num,
                                  std::int64_t row_len) {
  auto *const slice = createConstantFloat32(graph, {0.0}, {row_len});
  return createDynamicslice(graph, {value, row_num, slice->output()}, {0}, {1},
                            1);
}

torch::jit::Node *prepareArgmax(torch::jit::Graph *graph,
                                torch::jit::Value *const dists,
                                torch::jit::Value *const row_num,
                                std::int64_t row_len) {
  auto *const dists_row = prepareRowSlice(graph, dists, row_num, row_len);
  return createArgmax(graph, {dists_row->output()}, 0, 0l);
}

torch::jit::Node *prepareOutput(torch::jit::Graph *graph,
                                std::int64_t src_len) {
  const auto zeros = std::vector<std::int64_t>(src_len, 0l);
  return createConstantInt(graph, zeros, {src_len});
}

torch::jit::Node *prepareStartIdx(torch::jit::Graph *graph, float range_begin,
                                  float range_end, bool random_start) {
  if (random_start) {
    return createRandomUniform(graph, nullptr, {1}, range_end, range_begin,
                               c10::ScalarType::Int);
  }
  return createConstantLong(graph, {static_cast<std::int64_t>(range_begin)},
                            {1});
}

torch::jit::Node *prepareMaskedColDists(torch::jit::Graph *graph,
                                        torch::jit::Value *const dists,
                                        torch::jit::Value *const col_idx) {
  return createDynamiczero(graph, {dists, col_idx}, {1}, {1});
}

torch::jit::Node *prepareDists(torch::jit::Graph *graph,
                               torch::jit::Value *const src) {
  auto *const p = createConstantFloat32(graph, {2.0}, {1});
  return createHandlerOperation(graph, getHandler(c10::aten::cdist),
                                {src, src, p->output()});
}

torch::jit::Node *maskDists(torch::jit::Graph *graph,
                            torch::jit::Value *const dists,
                            const std::vector<std::int64_t> &offset,
                            const std::vector<std::int64_t> &sizes) {
  auto *const offset_node = createConstantInt(graph, offset, {2});
  return createDynamiczero(graph, {dists, offset_node->output()}, {0, 1},
                           sizes);
}

torch::jit::Node *prepareMaskedDists(torch::jit::Graph *graph,
                                     torch::jit::Value *const src,
                                     const std::vector<std::int64_t> &ptr) {
  auto *dists = prepareDists(graph, src);
  if (ptr.size() > 2) {
    dists = maskDists(graph, dists->output(), {0, ptr[1]},
                      {ptr[1], ptr.back() - ptr[1]});
    for (size_t i = 2; i < ptr.size() - 1; i++) {
      dists = maskDists(graph, dists->output(), {ptr[i - 1], 0},
                        {ptr[i] - ptr[i - 1], ptr[i - 1]});
      dists = maskDists(graph, dists->output(), {ptr[i - 1], ptr[i]},
                        {ptr[i] - ptr[i - 1], ptr.back() - ptr[i]});
    }
    dists = maskDists(graph, dists->output(), {ptr[ptr.size() - 2], 0},
                      {ptr.back() - ptr[ptr.size() - 2], ptr[ptr.size() - 2]});
  }
  return dists;
}

std::vector<std::int64_t> calcDeg(const std::vector<std::int64_t> &ptr,
                                  float ratio) {
  std::vector<std::int64_t> deg(ptr.size(), 0);
  for (size_t i = 1; i < ptr.size(); i++) {
    deg[i] = std::ceil(static_cast<float>(ptr[i] - ptr[i - 1]) * ratio);
    deg[i] += deg[i - 1];
  }
  return deg;
}

[[maybe_unused]] torch::jit::Node *
fpsHandler(torch::jit::Graph *graph, [[maybe_unused]] torch::jit::Node *node) {
  torch::jit::Value *const src = node->input(0);

  const std::vector<std::int64_t> ptr =
      constantToLongVec(node->input(1)->node());
  const float ratio = constantToFloat(node->input(2)->node());
  const bool random_start = constantToBool(node->input(3)->node());

  const std::vector<std::int64_t> src_shape = shapeFromTensor(src);

  // 0. Prepare output tensor
  const auto deg = calcDeg(ptr, ratio);
  const auto out_len = deg.back();
  auto *idxs = prepareOutput(graph, out_len);

  // 1. Create masked dists (leave only the slices representing batches)
  auto *dists = prepareMaskedDists(graph, src, ptr);

  // 2. Iterate over batches defined in deg
  std::int64_t pos_in_idxs = 0;
  for (size_t b = 1; b < deg.size(); b++) {
    // 3. Generate start idx...
    auto *prev_idx =
        prepareStartIdx(graph, ptr[b - 1], ptr[b] - 1, random_start);

    // 4. ...and insert it into the outputs
    idxs = updateIdxs(graph, idxs->output(), pos_in_idxs++, prev_idx->output());
    if (pos_in_idxs == deg[b] || pos_in_idxs == out_len) {
      continue;
    }

    // 5. Zero out the dists column with prev_idx number
    dists = prepareMaskedColDists(graph, dists->output(), prev_idx->output());

    // 6. Get the index of the max value in the currently processed dists row
    auto *idx =
        prepareArgmax(graph, dists->output(), prev_idx->output(), src_shape[0]);
    idxs = updateIdxs(graph, idxs->output(), pos_in_idxs++, idx->output());

    while (pos_in_idxs < deg[b] && pos_in_idxs < out_len) {
      // 7. Zero out the dists column with idx number
      dists = prepareMaskedColDists(graph, dists->output(), idx->output());
      auto *const prev_row = prepareRowSlice(graph, dists->output(),
                                             prev_idx->output(), src_shape[0]);
      auto *const curr_row =
          prepareRowSlice(graph, dists->output(), idx->output(), src_shape[0]);

      // 8. Update the currently processed row with the min of the current and
      // previous row
      auto *const curr_dists_row =
          prepareMinimum(graph, prev_row->output(), curr_row->output());
      dists = prepareRowUpdate(graph, dists->output(), curr_dists_row->output(),
                               idx->output(), src_shape[0]);

      prev_idx = idx;
      // 9. Get the index of the max value in the currently processed dists row
      idx = prepareArgmax(graph, dists->output(), idx->output(), src_shape[0]);
      idxs = updateIdxs(graph, idxs->output(), pos_in_idxs++, idx->output());
    }
  }
  return idxs;
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(poptorch::symbols::poptorch::fps, fpsHandler);
}

} // namespace poptorch
