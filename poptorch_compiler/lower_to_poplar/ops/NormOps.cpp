// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <popnn/BatchNorm.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

#include "lower_to_poplar/CompilerHelpers.hpp"

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

void batch_norm::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor weight;
  poplar::Tensor bias;
  poplar::Tensor running_mean;
  poplar::Tensor running_var;
  bool training = this->training();
  float momentum = this->momentum().convertToFloat();
  float epsilon = this->epsilon().convertToFloat();

  std::uint64_t c = input.shape()[1];
  std::vector<float> ones(c, 1.0f);
  std::vector<float> zeros(c, 0.0f);
  std::vector<uint64_t> param_shape = {c};
  if (this->weight() && this->bias()) {
    weight = context.fromSsa(this->weight());
    bias = context.fromSsa(this->bias());
  } else {
    weight = createConstant(context, poplar::FLOAT, param_shape, ones);
    bias = createConstant(context, poplar::FLOAT, param_shape, zeros);
  }
  if (this->running_mean() && this->running_var()) {
    running_mean = context.fromSsa(this->running_mean());
    running_var = context.fromSsa(this->running_var());
  } else {
    running_mean = createConstant(context, poplar::FLOAT, param_shape, zeros);
    running_var = createConstant(context, poplar::FLOAT, param_shape, ones);
  }

  if (training) {
    poplar::Tensor batch_mean;
    poplar::Tensor inv_sd;
    std::tie(batch_mean, inv_sd) = popnn::bn::batchNormStatistics(
        context.graph, input, epsilon, context.seq, false,
        true /* stable_algo */, poplar::FLOAT);

    // batch normalise
    context.tensors[this->result()] =
        batchNormalise(context, input, weight, bias, batch_mean, inv_sd);

    // Ensure batch mean is the same type as mean so that running mean can
    // be calculated
    if (batch_mean.elementType() != running_mean.elementType()) {
      batch_mean = popops::cast(context.graph, batch_mean,
                                running_mean.elementType(), context.seq);
    }

    // Then convert the inv_sd to the variance
    auto batch_var = popops::invStdDevToVariance(
        context.graph, inv_sd, epsilon, context.seq, running_var.elementType());

    // Calculate the running mean
    context.tensors[this->mean()] = popops::map(
        context.graph,
        pe::Add(pe::Mul(pe::Sub(pe::Const(1), pe::Const(momentum)), pe::_2),
                pe::Mul(pe::Const(momentum), pe::_1)),
        {running_mean, batch_mean}, context.seq);

    // Calculate the running variance using the unbiased results
    context.tensors[this->var()] = popops::map(
        context.graph,
        pe::Add(pe::Mul(pe::Sub(pe::Const(1), pe::Const(momentum)), pe::_2),
                pe::Mul(pe::Const(momentum), pe::_1)),
        {running_var, batch_var}, context.seq);
  } else {
    // convert variance to inverse standard deviation
    auto inv_sd = popops::varianceToInvStdDev(
        context.graph, running_var, epsilon, context.seq, input.elementType());

    // mean might have a different type so cast is required before
    // batchNormalise calculation
    if (running_mean.elementType() != input.elementType()) {
      running_mean = popops::cast(context.graph, running_mean,
                                  input.elementType(), context.seq);
    }

    // batchnorm
    context.tensors[this->result()] =
        batchNormalise(context, input, weight, bias, running_mean, inv_sd);
  }
}

} // namespace poptorch_ir
