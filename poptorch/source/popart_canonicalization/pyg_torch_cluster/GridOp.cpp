// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "../PopartCanonicalizationUtils.hpp"
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"

#include "../ScatterReduction.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

torch::jit::Node *gridHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto *pos = node->input(0);
  auto *size = node->input(1);

  auto *start = node->input(2);
  auto *end = node->input(3);

  std::vector<std::int64_t> pos_shape = shapeFromTensor(pos);
  const std::vector<std::int64_t> size_shape = shapeFromTensor(size);

  int num_voxels_size = 1;

  if (pos_shape.size() > 1) {
    num_voxels_size = std::accumulate(pos_shape.cbegin() + 1, pos_shape.cend(),
                                      1, std::multiplies<int>());
    pos_shape = {pos_shape[0], num_voxels_size};
    pos = createReshape(graph, pos, pos_shape)->output();
  }

  if (isNone(start)) {
    start = createReducemin(graph, {pos}, {0}, 0)->output();
  }

  if (isNone(end)) {
    end = createReducemax(graph, {pos}, {0}, 0)->output();
  }
  pos = createSub(graph, {pos, createUnsqueeze(graph, {start}, {0})->output()})
            ->output();

  start = createCast(graph, start, c10::kFloat)->output();
  end = createCast(graph, end, c10::kFloat)->output();
  size = createCast(graph, size, c10::kFloat)->output();

  auto *ones = wrapInConstantVec(graph, {1});
  auto *zeros = wrapInConstantVec(graph, {0});

  auto *num_voxels =
      createDiv(graph, {createSub(graph, {end, start})->output(), size})
          ->output();
  num_voxels = createCast(graph, num_voxels, c10::kInt)->output();
  num_voxels =
      createAdd(graph, {num_voxels, wrapInConstantVec(graph, {1})})->output();
  num_voxels->setType(num_voxels->type()->expect<c10::TensorType>()->withSizes(
      {num_voxels_size}));
  num_voxels = createHandlerOperation(graph, getHandler(c10::aten::cumprod),
                                      {num_voxels, zeros})
                   ->output();
  num_voxels = createConcat(graph, {ones, num_voxels}, 0)->output();
  num_voxels =
      createSlice(graph, {num_voxels}, {size_shape.at(0)}, {0}, {0})->output();

  num_voxels->setType(num_voxels->type()->expect<c10::TensorType>()->withSizes(
      {size_shape.at(0)}));

  pos = createCast(graph, pos, c10::kFloat)->output();
  size =
      createReshape(graph, size,
                    {1, std::accumulate(size_shape.cbegin(), size_shape.cend(),
                                        1, std::multiplies<int>())})
          ->output();
  auto *out = createDiv(graph, {pos, size})->output();
  out = createCast(graph, out, c10::kInt)->output();
  out = createMul(graph, {out, num_voxels})->output();

  return createReducesum(graph, {out}, {1}, 0);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(torch_cluster::grid, gridHandler);
}

} // namespace poptorch
