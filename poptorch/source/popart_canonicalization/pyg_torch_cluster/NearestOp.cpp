// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <random>

#include <ATen/ExpandUtils.h>

#include "../PopartCanonicalizationUtils.hpp"
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

std::tuple<std::int64_t, std::int64_t>
batchSizes(const torch::jit::Value *x, const torch::jit::Value *y,
           std::vector<std::int64_t> &batch_x,
           std::vector<std::int64_t> &batch_y) {
  if (!std::is_sorted(batch_x.cbegin(), batch_x.cend())) {
    throw std::invalid_argument("'batch_x' is not sorted");
  }
  if (!std::is_sorted(batch_y.cbegin(), batch_y.cend())) {
    throw std::invalid_argument("'batch_y' is not sorted");
  }

  std::int64_t batch_x_size = batch_x.size();
  std::int64_t batch_y_size = batch_y.size();

  if (batch_x_size == 0 && (batch_y_size != 0)) {
    batch_x_size = shapeFromTensor(x)[0];
    batch_x = std::vector<std::int64_t>(batch_x_size, 0);
  }

  if (batch_y_size == 0 && (batch_x_size != 0)) {
    batch_y_size = shapeFromTensor(y)[0];
    batch_y = std::vector<std::int64_t>(batch_y_size, 0);
  }

  return {batch_x_size, batch_y_size};
}

std::tuple<std::vector<std::int64_t>, std::vector<std::int64_t>>
batchShapes(torch::jit::Graph *graph, const std::vector<std::int64_t> &x_shape,
            const std::vector<std::int64_t> &y_shape,
            torch::jit::Value *&batch_x, torch::jit::Value *&batch_y) {
  std::vector<std::int64_t> batch_x_shape = shapeFromTensor(batch_x);
  std::vector<std::int64_t> batch_y_shape = shapeFromTensor(batch_y);

  if (batch_x_shape[0] == 0 && (batch_y_shape[0] != 0)) {
    batch_x_shape = {x_shape[0]};
    const std::vector<std::int64_t> data(batch_x_shape[0], 0);
    batch_x = createConstantLong(graph, data, batch_x_shape)->output();
  }
  if (batch_y_shape[0] == 0 && (batch_x_shape[0] != 0)) {
    batch_y_shape = {y_shape[0]};
    const std::vector<std::int64_t> data(batch_y_shape[0], 0);
    batch_y = createConstantLong(graph, data, batch_y_shape)->output();
  }

  return {batch_x_shape, batch_y_shape};
}

std::vector<std::int64_t> prepareInputTensor(torch::jit::Graph *graph,
                                             torch::jit::Value *&input) {
  auto input_shape = shapeFromTensor(input);
  if (input_shape.size() == 1) {
    const auto input_accum_shape = std::accumulate(
        input_shape.cbegin(), input_shape.cend(), 1, std::multiplies<size_t>());
    input_shape = std::vector<std::int64_t>{input_accum_shape, 1};
    input = createReshape(graph, input, input_shape)->output();
  }

  return input_shape;
}

void validateInputShapes(const std::vector<std::int64_t> &x_shape,
                         const std::vector<std::int64_t> &y_shape) {
  if (x_shape.size() > 2 || y_shape.size() > 2) {
    throw std::invalid_argument(
        "Inputs `x` and `y` should be max 2D tensors, while `x` has " +
        std::to_string(x_shape.size()) + " dims and `y` has " +
        std::to_string(y_shape.size()) + " dims.");
  }
  if (x_shape[1] != y_shape[1]) {
    throw std::invalid_argument(
        "Inputs shapes inconsistent x.shape[1]=" + std::to_string(x_shape[1]) +
        " vs. y.shape[1]=" + std::to_string(y_shape[1]));
  }
}

std::vector<std::int64_t> uniqueConsecutive(std::vector<std::int64_t> batch) {
  auto last = std::unique(batch.begin(), batch.end());
  batch.erase(last, batch.end());
  return batch;
}

void validateBatchIndices(const std::vector<std::int64_t> &batch_x,
                          const std::vector<std::int64_t> &batch_y) {
  const auto unique_batch_x = uniqueConsecutive(batch_x);
  const auto unique_batch_y = uniqueConsecutive(batch_y);

  if (unique_batch_x != unique_batch_y) {
    throw std::invalid_argument("Some batch indices occur in 'batch_x' "
                                "that do not occur in 'batch_y'");
  }
}

void validateSizes(std::int64_t x_size, std::int64_t y_size,
                   std::int64_t batch_x_size, std::int64_t batch_y_size) {
  if (x_size != batch_x_size) {
    throw std::invalid_argument("x.size(0) == batch_x.size(0)");
  }
  if (y_size != batch_y_size) {
    throw std::invalid_argument("y.size(0) == batch_y.size(0)");
  }
}

void validateShapes(const std::vector<std::int64_t> &x_shape,
                    const std::vector<std::int64_t> &y_shape,
                    const std::vector<std::int64_t> &batch_x_shape,
                    const std::vector<std::int64_t> &batch_y_shape) {
  if (batch_x_shape.size() != 1 || x_shape.front() != batch_x_shape.front()) {
    throw std::invalid_argument("x.size(0) == batch_x.size(0)");
  }
  if (batch_y_shape.size() != 1 || y_shape.front() != batch_y_shape.front()) {
    throw std::invalid_argument("y.size(0) == batch_y.size(0)");
  }
}

void rescaleInputs(torch::jit::Graph *graph, torch::jit::Value *&x,
                   torch::jit::Value *&y,
                   const std::vector<std::int64_t> &x_shape,
                   const std::vector<std::int64_t> &y_shape) {
  static constexpr bool keepdims = false;
  torch::jit::Value *const min_x =
      createReducemin(graph, {x}, {0, 1}, static_cast<int64_t>(keepdims))
          ->output();
  torch::jit::Value *const min_y =
      createReducemin(graph, {y}, {0, 1}, static_cast<int64_t>(keepdims))
          ->output();
  torch::jit::Value *const min_xy = createMin(graph, {min_x, min_y})->output();
  x = createSub(graph, {x, min_xy})->output();
  y = createSub(graph, {y, min_xy})->output();

  torch::jit::Value *const max_x =
      createReducemax(graph, {x}, {0, 1}, static_cast<int64_t>(keepdims))
          ->output();
  torch::jit::Value *const max_y =
      createReducemax(graph, {y}, {0, 1}, static_cast<int64_t>(keepdims))
          ->output();
  torch::jit::Value *const max_xy = createMax(graph, {max_x, max_y})->output();

  x = createDiv(graph, {x, max_xy})->output();
  x->setType(x->type()->expect<c10::TensorType>()->withSizes(x_shape));
  y = createDiv(graph, {y, max_xy})->output();
  y->setType(y->type()->expect<c10::TensorType>()->withSizes(y_shape));
}

void concatFeatures(torch::jit::Graph *graph, torch::jit::Value *&input,
                    const std::vector<std::int64_t> &input_shape,
                    std::vector<std::int64_t> &batch, std::int64_t D) {
  std::transform(batch.cbegin(), batch.cend(), batch.begin(),
                 [&D](std::int64_t value) { return 2 * D * value; });
  torch::jit::Value *batch_tensor =
      createConstantLong(graph, batch,
                         {static_cast<std::int64_t>(batch.size()), 1})
          ->output();
  input = createConcat(graph, {input, batch_tensor}, 1)->output();
  const std::vector<std::int64_t> concat_shape{input_shape[0],
                                               input_shape[1] + 1};
  input->setType(
      input->type()->expect<c10::TensorType>()->withSizes(concat_shape));
}

void concatFeatures(torch::jit::Graph *graph, torch::jit::Value *&input,
                    const std::vector<std::int64_t> &input_shape,
                    torch::jit::Value *&batch, std::int64_t batch_size,
                    std::int64_t D) {
  const std::vector<std::int64_t> data(batch_size, 2 * D);
  const std::vector<std::int64_t> batch_shape{batch_size, 1};
  torch::jit::Value *multiplier =
      createConstantInt(graph, data, batch_shape)->output();
  batch = createReshape(graph, batch, batch_shape)->output();
  batch = createMul(graph, {multiplier, batch})->output();
  input = createConcat(graph, {input, batch}, 1)->output();
  const std::vector<std::int64_t> concat_shape{input_shape[0],
                                               input_shape[1] + 1};
  input->setType(
      input->type()->expect<c10::TensorType>()->withSizes(concat_shape));
}

torch::jit::Node *vq(torch::jit::Graph *graph, torch::jit::Value *const x,
                     torch::jit::Value *const y) {
  auto *const p = createConstantFloat32(graph, {2.0}, {1});
  auto *const distances = createHandlerOperation(
      graph, getHandler(c10::aten::cdist), {x, y, p->output()});
  return createArgmin(graph, {distances->output()}, 1 /*axis*/, 0 /*keepdims*/);
}

torch::jit::Node *nearestBatchListHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  torch::jit::Value *x = node->input(0);
  torch::jit::Value *y = node->input(1);

  std::vector<std::int64_t> batch_x = constantToLongVec(node->input(2)->node());
  std::vector<std::int64_t> batch_y = constantToLongVec(node->input(3)->node());

  const auto x_shape = prepareInputTensor(graph, x);
  const auto y_shape = prepareInputTensor(graph, y);

  validateInputShapes(x_shape, y_shape);

  const auto [batch_x_size, batch_y_size] = batchSizes(x, y, batch_x, batch_y);

  if ((batch_x_size != 0) && (batch_y_size != 0)) {
    validateBatchIndices(batch_x, batch_y);
    validateSizes(x_shape[0], y_shape[0], batch_x_size, batch_y_size);

    rescaleInputs(graph, x, y, x_shape, y_shape);

    const std::int64_t d = x_shape.back();
    concatFeatures(graph, x, x_shape, batch_x, d);
    concatFeatures(graph, y, y_shape, batch_y, d);
  }

  return vq(graph, x, y);
}

torch::jit::Node *nearestHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  torch::jit::Value *x = node->input(0);
  torch::jit::Value *y = node->input(1);
  torch::jit::Value *batch_x = node->input(2);
  torch::jit::Value *batch_y = node->input(3);

  const auto x_shape = prepareInputTensor(graph, x);
  const auto y_shape = prepareInputTensor(graph, y);

  validateInputShapes(x_shape, y_shape);

  const auto [batch_x_shape, batch_y_shape] =
      batchShapes(graph, x_shape, y_shape, batch_x, batch_y);

  if (!batch_x_shape.empty() && !batch_y_shape.empty()) {
    // No validation of batch indices as we can't assert from Poplar
    validateShapes(x_shape, y_shape, batch_x_shape, batch_y_shape);

    rescaleInputs(graph, x, y, x_shape, y_shape);

    const std::int64_t d = x_shape.back();
    concatFeatures(graph, x, x_shape, batch_x, batch_x_shape[0], d);
    concatFeatures(graph, y, y_shape, batch_y, batch_y_shape[0], d);
  }

  return vq(graph, x, y);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(poptorch::symbols::poptorch::nearest, nearestHandler);
  registerHandler(poptorch::symbols::poptorch::nearest_batch_list,
                  nearestBatchListHandler);
}

} // namespace poptorch
