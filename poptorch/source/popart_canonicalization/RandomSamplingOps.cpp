// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>

#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *normalHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // Overloads for aten::normal
  // 1) both mean and std are scalar floats
  //   aten::normal(float mean, float std, int[] size, Generator?, int? dtype,
  //   int? layout, Device? device, bool? pin_memory) -> Tensor
  //
  // 2) mean is a tensor and std is a scalar
  //   aten::normal(Tensor mean, float std, Tensor? out)
  //
  // 3) mean is a scalar and std is a tensor
  //   aten::normal(float mean, Tensor std, Tensor? out)
  //
  // 4) both mean and std are tensors
  //   aten::normal(Tensor mean, Tensor std, Tensor? out)
  torch::jit::Value *mean = node->input(0);
  torch::jit::Value *std = node->input(1);
  std::vector<int64_t> shape = shapeFromTensor(node->output());

  bool mean_scalar = isConstantScalar(mean);
  bool std_scalar = isConstantScalar(std);
  if (mean_scalar && std_scalar) {
    // Both mean and std are scalar constant floats
    float mean_constant = constantToFloat(mean->node());
    float std_constant = constantToFloat(std->node());

    return createRandomNormal(graph, {mean, std}, shape, mean_constant,
                              std_constant);
  }

  // One or both of mean/std inputs must be tensors.  Generate the output tensor
  // of random numbers drawn from separate normal distribution whose mean and
  // std are given as tensors using the following transform:
  //
  //   normal(mean=0, std=1) * std + mean
  //
  // Broadcasting will take care of expanding any scalars to the correct shape.
  // Use {mean} to identify the type only
  auto mean_type = getNodeScalarType(mean);
  auto std_type = getNodeScalarType(std);
  if (mean_type != std_type) {
    if (mean_scalar && !std_scalar) {
      mean = createCast(graph, mean, std_type)->output();
    }
    if (!mean_scalar && std_scalar) {
      std = createCast(graph, std, mean_type)->output();
    }
  }
  torch::jit::Node *normal =
      createRandomNormal(graph, {mean, std}, shape, 0.0f, 1.0f);
  torch::jit::Node *mul = poptorch::createMul(graph, {normal->output(), std});
  return poptorch::createAdd(graph, {mul->output(), mean});
}

torch::jit::Node *bernoulliHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::bernoulli(Tensor self, float? probability)
  // Check for scalar probability
  torch::jit::Value *prob = node->input(1);

  if (isNone(prob)) {
    // probabilities passed as input tensor
    prob = node->input(0);
  }

  std::vector<int64_t> shape = shapeFromTensor(node->output());
  c10::ScalarType dtype = getNodeScalarType(node->input(0));

  torch::jit::Value *uniform =
      createRandomUniform(graph, nullptr, shape, 1.0, 0.0, dtype)->output();

  torch::jit::Value *lt = createLess(graph, {uniform, prob})->output();
  return createCast(graph, lt, dtype);
}

torch::jit::Node *exponentialHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // aten::exponential_(Tensor self, double lambda)
  torch::jit::Value *self = node->input(0);
  torch::jit::Value *lambda = node->input(1);
  torch::jit::Value *output = node->output();

  std::vector<int64_t> shape = shapeFromTensor(output);
  c10::ScalarType dtype = getNodeScalarType(self);
  c10::ScalarType dtype_rng = c10::ScalarType::Float;

  // Use smallest non-zero value to prevent the posibility of
  // log(0) with minimal bias on the sampling distribution
  float low = std::numeric_limits<float>::min();
  torch::jit::Value *x =
      createRandomUniform(graph, nullptr, shape, 1.0, low, dtype_rng)->output();

  auto *log_x = createLog(graph, {x})->output();
  auto *neg_log_x = createNeg(graph, {log_x})->output();
  auto *exponential = createDiv(graph, {neg_log_x, lambda})->output();
  return createCast(graph, exponential, dtype);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::normal, normalHandler);
  registerHandler(c10::aten::bernoulli, bernoulliHandler);
  registerHandler(c10::aten::exponential_, exponentialHandler);
}

} // namespace poptorch
