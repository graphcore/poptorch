// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

bool isConstantScalar(torch::jit::Value *input) {
  if (input->node()->kind() != symbols::poptorch::tensor_constant) {
    return false;
  }

  std::vector<int64_t> shape = shapeFromTensor(input);
  int64_t numel = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<int64_t>());

  return numel == 1;
}

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
  at::ScalarType dtype = getNodeScalarType(node->output());

  if (isConstantScalar(mean) && isConstantScalar(std)) {
    // Both mean and std are scalar constant floats
    float mean_constant = constantToFloat(mean->node());
    float std_constant = constantToFloat(std->node());
    return createRandomNormal(graph, shape, mean_constant, std_constant, dtype);
  }

  // One or both of mean/std inputs must be tensors.  Generate the output tensor
  // of random numbers drawn from separate normal distribution whose mean and
  // std are given as tensors using the following transform:
  //
  //   normal(mean=0, std=1) * std + mean
  //
  // Broadcasting will take care of expanding any scalars to the correct shape.
  torch::jit::Node *normal =
      createRandomNormal(graph, shape, 0.0f, 1.0f, dtype);
  torch::jit::Node *mul = poptorch::createMul(graph, {normal->output(), std});
  return poptorch::createAdd(graph, {mul->output(), mean});
}

} // namespace

static bool handler = registerHandlers(c10::aten::normal, normalHandler);

} // namespace poptorch
