// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "poptorch_logging/Error.hpp"

namespace pe = popops::expr;

namespace poptorch_ir {

poplar::Tensor
batchNormalise(CompilerContext &context, const poplar::Tensor &input,
               const poplar::Tensor &weight, const poplar::Tensor &bias,
               const poplar::Tensor &mean, const poplar::Tensor &inv_sd) {
  //  combinedMultiplicand = gamma / sDev
  //                       = gamma * inv_sd
  auto multiplcand = popops::map(context.graph, pe::Mul(pe::_1, pe::_2),
                                 {weight, inv_sd}, context.seq);

  // addend = beta - gamma * mean / sdDev
  //        = beta - gamma * mean * inv_sd
  auto addend =
      popops::map(context.graph, pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3)),
                  {bias, multiplcand, mean}, context.seq);

  // Perform the batchNorm
  return popnn::bn::batchNormalise(context.graph, input, multiplcand, addend,
                                   context.seq);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor> batchNormaliseGrad(
    CompilerContext &context, const poplar::Tensor &input,
    const poplar::Tensor &weight, const poplar::Tensor &save_mean,
    const poplar::Tensor &inv_sd, const poplar::Tensor &grad_out) {

  poplar::Tensor input_whitened = popnn::bn::batchNormWhiten(
      context.graph, input, save_mean, inv_sd, context.seq);

  // Compute the delta for the operand
  poplar::Tensor grad_input =
      popnn::bn::batchNormGradients(context.graph, input_whitened, grad_out,
                                    inv_sd, weight, context.seq, poplar::FLOAT);

  const auto [grad_weight, grad_bias] = // NOLINT
      popnn::bn::batchNormParamGradients(context.graph, input_whitened,
                                         grad_out, context.seq, poplar::FLOAT);

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

void batch_norm::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor weight;
  poplar::Tensor bias;
  poplar::Tensor running_mean;
  poplar::Tensor running_var;
  bool training = this->training();
  float momentum = this->momentum().convertToFloat();
  float epsilon = this->epsilon().convertToFloat();

  std::vector<uint64_t> param_shape = {input.shape()[1]};
  if (this->weight() && this->bias()) {
    weight = context.fromSsa(this->weight());
    bias = context.fromSsa(this->bias());
  } else {
    weight = createConstant(context, poplar::FLOAT, param_shape, 1.0f);
    bias = createConstant(context, poplar::FLOAT, param_shape, 0.0f);
  }
  const bool track_running_stats = this->running_mean() && this->running_var();
  if (track_running_stats) {
    running_mean = context.fromSsa(this->running_mean());
    running_var = context.fromSsa(this->running_var());
  } else {
    running_mean = createConstant(context, poplar::FLOAT, param_shape, 0.0f);
    running_var = createConstant(context, poplar::FLOAT, param_shape, 1.0f);
  }

  if (training) {
    const auto [batch_mean, inv_sd] = popnn::bn::batchNormStatistics( // NOLINT
        context.graph, input, epsilon, context.seq,
        /*unbiasedVarEstimate=*/false, /*stableAlgo=*/true, poplar::FLOAT);

    context.addTensor(this->result(), batchNormalise(context, input, weight,
                                                     bias, batch_mean, inv_sd));

    // Save the computed mean and invstd for the backward op to reuse
    context.addTensor(this->save_mean(), batch_mean);
    context.addTensor(this->save_invstd(), inv_sd);

    if (track_running_stats) {
      // Calculate the running mean
      context.addTensor(
          this->running_mean(),
          popops::map(context.graph,
                      pe::Add(pe::Mul(pe::Const(1.f - momentum), pe::_1),
                              pe::Mul(pe::Const(momentum), pe::_2)),
                      {running_mean, batch_mean}, context.seq),
          /*update_if_present=*/true);

      auto batch_var =
          popops::invStdDevToVariance(context.graph, inv_sd, epsilon,
                                      context.seq, running_var.elementType());

      // Calculate the running variance
      context.addTensor(
          this->running_var(),
          popops::map(context.graph,
                      pe::Add(pe::Mul(pe::Const(1.f - momentum), pe::_1),
                              pe::Mul(pe::Const(momentum), pe::_2)),
                      {running_var, batch_var}, context.seq),
          /*update_if_present=*/true);
    }
  } else {
    auto inv_sd = popops::varianceToInvStdDev(
        context.graph, running_var, epsilon, context.seq, input.elementType());

    context.addTensor(
        this->result(),
        batchNormalise(context, input, weight, bias, running_mean, inv_sd));
  }
}

void batch_norm_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor grad_out = context.fromSsa(this->grad_out());
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor weight = context.fromSsa(this->weight());
  bool training = this->training();
  float epsilon = this->epsilon().convertToFloat();

  poplar::Tensor mean;
  poplar::Tensor inv_std;
  if (training) {
    // These tensors only exist in training mode, and are cached versions
    // of the batch mean and batch var tensors which are carried over from
    // the forward pass to avoid recomputing them for the backward pass
    mean = context.fromSsa(this->save_mean());
    inv_std = context.fromSsa(this->save_invstd());
  } else {
    mean = context.fromSsa(this->running_mean());
    poplar::Tensor running_var = context.fromSsa(this->running_var());
    inv_std = popops::varianceToInvStdDev(context.graph, running_var, epsilon,
                                          context.seq, input.elementType());
  }

  const auto [grad_input, grad_weight, grad_bias] = // NOLINT
      batchNormaliseGrad(context, input, weight, mean, inv_std, grad_out);

  context.addTensor(this->grad_input(), grad_input);
  context.addTensor(this->grad_weight(), grad_weight);
  context.addTensor(this->grad_bias(), grad_bias);
}

void group_norm::lowerToPoplar(CompilerContext &context) {
  float epsilon = this->epsilon().convertToFloat();
  uint64_t num_groups = this->group();

  // Hard wire to stable for now
  const bool stable_algo = true;

  // Get the inputs
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor weight;
  poplar::Tensor bias;

  // Check that the redundant N, C and HxW match input dimensions
  auto input_shape = input.shape();
  ERROR_ON(input_shape.at(0) != this->N());
  ERROR_ON(input_shape.at(1) != this->C());
  auto hx_w =
      std::accumulate(input_shape.begin() + 2, input_shape.end(),
                      static_cast<size_t>(1), std::multiplies<size_t>());
  ERROR_ON(hx_w != this->HxW());

  const std::vector<uint64_t> param_shape = {num_groups};
  if (this->weight() && this->bias()) {
    weight = context.fromSsa(this->weight());
    bias = context.fromSsa(this->bias());
  } else {
    weight = createConstant(context, poplar::FLOAT, param_shape, 1.0f);
    bias = createConstant(context, poplar::FLOAT, param_shape, 0.0f);
  }

  // Calculate the mean and the inverse standard deviation
  poplar::Tensor mean;
  poplar::Tensor inv_std_dev;

  // Hardwire to correct and slightly slower.
  const bool fast_math_group_norm = false;

  poplar::OptionFlags flags{{"groupNormStridedChannelGrouping",
                             fast_math_group_norm ? "true" : "false"}};

  std::tie(mean, inv_std_dev) = popnn::gn::groupNormStatistics(
      context.graph, input, epsilon, context.seq,
      static_cast<unsigned int>(num_groups), false, stable_algo, poplar::FLOAT,
      {}, flags);

  // Calculate the normalization
  auto result =
      popnn::gn::groupNormalise(context.graph, input, weight, bias, mean,
                                inv_std_dev, context.seq, {}, flags);

  // Return the result
  context.addTensor(this->result(), result.first);
  context.addTensor(this->mean(), mean);
  context.addTensor(this->rstd(), inv_std_dev);
}

} // namespace poptorch_ir
