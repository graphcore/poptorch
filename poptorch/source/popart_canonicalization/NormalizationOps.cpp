// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

// Ensures running_mean and running_var tensors by creating constants if they
// are not set (None) The running_mean and running_var may be none e.g. if
// track_running_stats is set to False for the relevant PyTorch BatchNorm layer.
// To satisfy popart/onnx, create a zero input for running_mean and all ones for
// running_var
void maybeInitializeRunningParamConstants(
    torch::jit::Graph *graph, const c10::ScalarType input_type,
    torch::jit::Value **running_mean, torch::jit::Value **running_var,
    const std::vector<std::int64_t> &new_shape) {

  std::vector<int64_t> running_shape{new_shape[1]};

  if (!isNone(*running_mean)) {
    ERROR_ON(isNone(*running_var));
    return;
  }

  ERROR_ON(!isNone(*running_var));

  switch (input_type) {
  case c10::ScalarType::Int: {
    *running_mean = createConstantInt(graph, {0}, running_shape)->output();
    *running_var = createConstantInt(graph, {1}, running_shape)->output();
    break;
  }
  case c10::ScalarType::Float: {
    *running_mean = createConstantFloat(graph, {0.0}, running_shape)->output();
    *running_var = createConstantFloat(graph, {1.0}, running_shape)->output();
    break;
  }
  case c10::ScalarType::Half: {
    *running_mean =
        createConstantFloat16(graph, {0.0}, running_shape)->output();
    *running_var = createConstantFloat16(graph, {1.0}, running_shape)->output();
    break;
  }
  default: {
    ERROR("Batch norm input"
          << " of type " << c10::toString(input_type) << " not supported");
  }
  }
}

// Flattens or expands input tensor to 4D and returns new 4D tensor
torch::jit::Value *get4dInput(torch::jit::Graph *graph,
                              torch::jit::Value *input) {
  std::vector<std::int64_t> shape = shapeFromTensor(input);
  if (shape.size() == 4) {
    ERROR("Input is already 4-dimensional");
  }

  // Turn the shape into a 4D tensor.
  if (shape.size() < 4) {
    // Pad to 4D with singletons.
    std::size_t padding = 4 - shape.size();
    std::generate_n(std::back_inserter(shape), padding, []() { return 1; });
  } else if (shape.size() > 4) {
    // Flatten excess dimensions to reduce to 4.
    shape[3] = std::accumulate(shape.begin() + 4, shape.end(), shape[3],
                               std::multiplies<std::int64_t>());
    shape.resize(4);
  }

  return createReshape(graph, input, shape)->output();
}

// Helper function used for normalization functions where input tensors are
// required to be 4D. Reshapes to 4D, performs the normalization function
// passed to it, and then reshapes the output back to its original shape
template <typename NormFunc>
torch::jit::Node *normalizeReshapeIfNeeded(torch::jit::Graph *graph,
                                           torch::jit::Value *input,
                                           NormFunc &&normalize_fn) {
  // Reshape to 4D if needed
  std::vector<std::int64_t> original_shape = shapeFromTensor(input);
  if (original_shape.size() != 4) {
    input = get4dInput(graph, input);
  }

  torch::jit::Node *new_node = normalize_fn(input);

  // If we reshaped, reshape back
  if (original_shape.size() != 4) {
    new_node = createReshape(graph, new_node->output(), original_shape);
  }

  return new_node;
}

torch::jit::Node *batchNormHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
  // running_mean, Tensor?  , bool training, float momentum, float
  // eps, bool cudnn_enabled) -> Tensor

  // Input is value at 0th position.
  torch::jit::Value *input = node->input(0);

  torch::jit::Value *weight = node->input(1);
  torch::jit::Value *bias = node->input(2);

  torch::jit::Value *running_mean = node->input(3);
  torch::jit::Value *running_var = node->input(4);

  float momentum = constantToFloat(node->input(6)->node());
  float epsilon = constantToFloat(node->input(7)->node());

  bool training = constantToBool(node->input(5)->node());

  auto batch_norm_fn = [&](torch::jit::Value *input_4d) {
    // Use initialised constants if running_mean and running_var are none
    maybeInitializeRunningParamConstants(
        graph, *input_4d->type()->expect<c10::TensorType>()->scalarType(),
        &running_mean, &running_var, shapeFromTensor(input_4d));

    // To indicate training, for BatchNormalization-9, use num_outputs = 5
    // From ONNX
    // Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
    // Output case #2: Y (test mode)
    // Popart supports this with "if (output->n() > 1)"
    return createBatchnormalization(
        graph, {input_4d, weight, bias, running_mean, running_var},
        training ? 5 : 1, epsilon, momentum);
  };

  // Pytorch supports BatchNorm1D/2D/3D. PopART only supports 2D so we need
  // to reshape input into a 4D tensor.
  return normalizeReshapeIfNeeded(graph, input, batch_norm_fn);
}

torch::jit::Node *layerNormHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::layer_norm(Tensor input,int[] normalized_shape, Tensor? weight,
  //                Tensor? bias, float eps, bool cudnn_enable) -> Tensor

  // Tensor to normalise.
  torch::jit::Value *input = node->input(0);

  // Weight to multiply.
  torch::jit::Value *gamma = node->input(2);

  // Bias to add.
  torch::jit::Value *beta = node->input(3);

  const float epsilon = constantToFloat(node->input(4)->node());

  // Pytorch normalizes across arbitrary number of dimensions from the end.
  // We flatten into a [M, N] array and normalize the N.

  std::vector<std::int64_t> output_shape = shapeFromTensor(node->output());
  std::vector<std::int64_t> normalized_shape =
      constantToLongVec(node->input(1)->node());
  std::vector<std::int64_t> input_shape = shapeFromTensor(input);
  const std::int64_t axis = input_shape.size() - normalized_shape.size();

  // Flatten into [M, N]
  torch::jit::Node *flatten = createFlatten(graph, {input}, axis);

  // Normalize.
  torch::jit::Node *normalize = createGroupnormalization(
      graph, {flatten->output(), gamma, beta}, 1, epsilon);

  // Perform the reshape.
  return createReshape(graph, normalize->output(), output_shape);
}

// This handler ensures that the input to popart is 4-dimensional
torch::jit::Node *groupNormHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor?
  //                  bias, float eps, bool cudnn_enabled)

  torch::jit::Value *input = node->input(0);

  std::int64_t num_groups = constantToLong(node->input(1)->node());
  // Weight to multiply
  torch::jit::Value *gamma = node->input(2);
  // Bias to add
  torch::jit::Value *beta = node->input(3);
  float epsilon = constantToFloat(node->input(4)->node());

  auto group_norm_fn = [&](torch::jit::Value *input_4d) {
    return createGroupnormalization(graph, {input_4d, gamma, beta}, num_groups,
                                    epsilon);
  };

  return normalizeReshapeIfNeeded(graph, input, group_norm_fn);
}

torch::jit::Node *instanceNormHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  // aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
  //                     running_mean, Tensor? running_var, bool
  //                     use_input_stats, float momentum, float eps, bool
  //                     cudnn_enabled)

  // Tensor to normalise
  // Input: (N, C, L)       InstanceNorm1d
  //        (N, C, H, W)    InstanceNorm2d
  //        (N, C, D, H, W) InstanceNorm3d
  torch::jit::Value *input = node->input(0);

  // Weight to multiply
  torch::jit::Value *gamma = node->input(1);

  // Bias to add
  torch::jit::Value *beta = node->input(2);

  // Group normalization does not currently allow passing a momentum value,
  // nor the running mean or running variance

  float epsilon = constantToFloat(node->input(7)->node());
  std::int64_t num_channels = shapeFromTensor(input)[1];

  auto instance_norm_fn = [&](torch::jit::Value *input_4d) {
    // Normalize per channel C, so use Group normalization with C groups
    return createGroupnormalization(graph, {input_4d, gamma, beta},
                                    num_channels, epsilon);
  };

  return normalizeReshapeIfNeeded(graph, input, instance_norm_fn);
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::batch_norm, batchNormHandler,
    c10::aten::layer_norm, layerNormHandler,
    c10::aten::group_norm, groupNormHandler,
    c10::aten::instance_norm, instanceNormHandler);
// clang-format on

} // namespace poptorch
