// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/ImplicitCasting.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

void initializeParamConstant(torch::jit::Graph *graph, torch::jit::Value *input,
                             torch::jit::Value **param, float value,
                             const std::vector<int64_t> &shape,
                             const std::string &norm_name,
                             const std::string &input_name,
                             bool always_f32 = false) {
  c10::ScalarType scalar_type =
      *input->type()->expect<c10::TensorType>()->scalarType();
  switch (scalar_type) {
  case c10::ScalarType::Int: {
    *param = createConstantInt(graph, {static_cast<int64_t>(value)}, shape)
                 ->output();
    break;
  }
  case c10::ScalarType::Half:
  case c10::ScalarType::Float: {
    if (always_f32) {
      *param = createConstantFloat32(graph, {value}, shape)->output();
    } else {
      *param = createConstantFloatLike(graph, input, {value}, shape)->output();
    }
    break;
  }
  default:
    ERROR(norm_name << " input \"" << input_name << "\""
                    << " of type " << c10::toString(scalar_type)
                    << " not supported");
  }
}

// Return true if parameters are initialised by this function, otherwise return
// false
bool maybeInitializeAffineParamConstants(torch::jit::Graph *graph,
                                         torch::jit::Value *input,
                                         torch::jit::Value **weight,
                                         torch::jit::Value **bias,
                                         const std::vector<std::int64_t> &shape,
                                         const std::string &norm_name) {
  // Either both should be defined, or neither
  ERROR_ON(isNone(*weight) != isNone(*bias));
  if (!isNone(*weight)) {
    return false;
  }

  initializeParamConstant(graph, input, weight, 1, shape, norm_name, "weight");
  initializeParamConstant(graph, input, bias, 0, shape, norm_name, "bias");
  return true;
}

// Ensures running_mean and running_var tensors by creating constants if they
// are not set (None) The running_mean and running_var may be none e.g. if
// track_running_stats is set to False for the relevant PyTorch BatchNorm layer.
// To satisfy popart/onnx, create a zero input for running_mean and all ones for
// running_var
void maybeInitializeRunningParamConstants(
    torch::jit::Graph *graph, torch::jit::Value *input,
    torch::jit::Value **running_mean, torch::jit::Value **running_var,
    const std::vector<std::int64_t> &shape) {
  // Either both should be defined, or neither
  ERROR_ON(isNone(*running_mean) != isNone(*running_var));
  if (!isNone(*running_mean)) {
    return;
  }

  std::string norm_name = "BatchNorm";
  bool always_f32 = runningStatisticsAlwaysFloat();
  initializeParamConstant(graph, input, running_mean, 0, shape, norm_name,
                          "running_mean", always_f32);
  initializeParamConstant(graph, input, running_var, 1, shape, norm_name,
                          "running_var", always_f32);
}

torch::jit::Node *batchNormHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
  // running_mean, Tensor? running_var, bool training, float momentum, float
  // eps, bool cudnn_enabled) -> Tensor

  // aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
  // running_mean, Tensor? running_var, bool training, float momentum, float
  // eps) -> (Tensor, Tensor, Tensor)

  // Input is value at 0th position.
  torch::jit::Value *input = node->input(0);

  auto input_shape = shapeFromTensor(input);

  torch::jit::Value *weight = node->input(1);
  torch::jit::Value *bias = node->input(2);

  torch::jit::Value *running_mean = node->input(3);
  torch::jit::Value *running_var = node->input(4);

  float momentum = constantToFloat(node->input(6)->node());
  float epsilon = constantToFloat(node->input(7)->node());

  if (runningStatisticsAlwaysFloat()) {
    auto fn_ensure_float = [&](torch::jit::Value *stat_tensor) {
      if (!isNone(stat_tensor)) {
        // make sure the running statistics tensor is of type float
        auto old_type = stat_tensor->type()->cast<c10::TensorType>();
        stat_tensor->setType(old_type->withScalarType(at::ScalarType::Float));
      }
    };
    fn_ensure_float(running_mean);
    fn_ensure_float(running_var);
  }

  bool training = constantToBool(node->input(5)->node());
  bool three_outputs = (node->kind() == c10::aten::native_batch_norm);

  std::vector<int64_t> param_shape{input_shape[1]};

  maybeInitializeAffineParamConstants(graph, input, &weight, &bias, param_shape,
                                      "BatchNorm");

  // Use initialised constants if running_mean and running_var are none
  maybeInitializeRunningParamConstants(graph, input, &running_mean,
                                       &running_var, param_shape);

  // PyTorch supports an input size of (N, C, *) but PopART requires the spatial
  // dimension, so we must ensure an input size of (N, C, L, *)
  if (input_shape.size() == 2) {
    input = createUnsqueeze(graph, {input}, {2})->output();
  }

  // To indicate training, for BatchNormalization-9, use num_outputs = 5
  // From ONNX
  // Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
  // Output case #2: Y (test mode)
  // Popart supports this with "if (output->n() > 1)"
  auto *batch_norm = createBatchnormalization(
      graph, {input, weight, bias, running_mean, running_var}, training ? 5 : 1,
      epsilon, 1.0f - momentum, training && three_outputs ? 3 : 1);

  // If the input size was of rank 2, we need to squeeze out the added dim
  if (input_shape.size() == 2) {
    batch_norm = createSqueeze(graph, {batch_norm->output(0)}, {2});
  }
  return batch_norm;
}

torch::jit::Node *layerNormHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::layer_norm(Tensor input,int[] normalized_shape, Tensor? weight,
  //                Tensor? bias, float eps, bool cudnn_enable) -> Tensor

  // Tensor to normalise.
  torch::jit::Value *input = node->input(0);

  std::vector<std::int64_t> normalized_shape =
      constantToLongVec(node->input(1)->node());

  // Weight to multiply.
  torch::jit::Value *gamma = node->input(2);
  // Bias to add.
  torch::jit::Value *beta = node->input(3);
  auto numel_affine =
      std::accumulate(normalized_shape.begin(), normalized_shape.end(), 1,
                      std::multiplies<int64_t>{});
  bool initialized = maybeInitializeAffineParamConstants(
      graph, input, &gamma, &beta, {numel_affine}, "LayerNorm");

  if (!initialized) {
    // GroupNorm takes per-channel affine parameters whereas LayerNorm takes
    // elementwise affine parameters. Therefore we first need to reshape such
    // that the affine parameters are "per-channel" which in the case of
    // LayerNorm is equivalent to flattening them
    gamma =
        createReshape(graph, gamma, {static_cast<std::int64_t>(numel_affine)})
            ->output();
    beta = createReshape(graph, beta, {static_cast<std::int64_t>(numel_affine)})
               ->output();
  }

  const float epsilon = constantToFloat(node->input(4)->node());

  // Pytorch normalizes across arbitrary number of dimensions from the end.
  // We flatten into a [M, N] array and normalize the N.
  std::vector<std::int64_t> output_shape = shapeFromTensor(node->output());
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

  auto num_channels = shapeFromTensor(input)[1];
  maybeInitializeAffineParamConstants(graph, input, &gamma, &beta,
                                      {num_channels}, "GroupNorm");

  float epsilon = constantToFloat(node->input(4)->node());

  return createGroupnormalization(graph, {input, gamma, beta}, num_groups,
                                  epsilon);
}

// aten::native_group_norm has a different signature to aten::group_norm
torch::jit::Node *nativeGroupNormHandler(torch::jit::Graph *graph,
                                         torch::jit::Node *node) {
  // aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N,
  // int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)

  // Returns are (result, mean, inv_std_dev) which matches PopTorch

  torch::jit::Value *input = node->input(0);

  // Weight to multiply
  torch::jit::Value *gamma = node->input(1);
  // Bias to add
  torch::jit::Value *beta = node->input(2);

  auto num_channels = shapeFromTensor(input)[1];
  maybeInitializeAffineParamConstants(graph, input, &gamma, &beta,
                                      {num_channels}, "GroupNorm");

  // N, C and HxW are redundant given that the input size must be known for
  // IPU, but provide a useful check
  auto input_shape = shapeFromTensor(input);
  ERROR_ON(input_shape[0] != constantToLong(node->input(3)->node()));
  ERROR_ON(input_shape[1] != constantToLong(node->input(4)->node()));

  auto hx_w =
      std::accumulate(input_shape.begin() + 2, input_shape.end(),
                      static_cast<int64_t>(1), std::multiplies<int64_t>());
  ERROR_ON(hx_w != constantToLong(node->input(5)->node()));

  std::int64_t num_groups = constantToLong(node->input(6)->node());

  float epsilon = constantToFloat(node->input(7)->node());
  return createGroupnormalization(graph, {input, gamma, beta}, num_groups,
                                  epsilon);
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

  std::int64_t num_channels = shapeFromTensor(input)[1];

  maybeInitializeAffineParamConstants(graph, input, &gamma, &beta,
                                      {num_channels}, "InstanceNorm");

  // Group normalization does not currently allow passing a momentum value,
  // nor the running mean or running variance

  float epsilon = constantToFloat(node->input(7)->node());

  // Normalize per channel C, so use Group normalization with C groups
  return createGroupnormalization(graph, {input, gamma, beta}, num_channels,
                                  epsilon);
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::batch_norm, batchNormHandler);
  registerHandler(c10::aten::native_batch_norm, batchNormHandler);
  registerHandler(c10::aten::layer_norm, layerNormHandler);
  registerHandler(c10::aten::group_norm, groupNormHandler);
  registerHandler(c10::aten::native_group_norm, nativeGroupNormHandler);
  registerHandler(c10::aten::instance_norm, instanceNormHandler);
}

} // namespace poptorch
