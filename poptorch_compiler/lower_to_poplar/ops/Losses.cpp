
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/CompilerHelpers.hpp"
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>

namespace pe = popops::expr;

namespace poptorch_ir {

void mse_loss_backward::lowerToPoplar(CompilerContext &context) {
  // aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int
  // reduction) -> (Tensor)
  //    alpha * (input - target) * grad_output;
  // Where alpha is
  //      2   when Reduction = Sum or None
  //      2 / numel()  when Reduction = Sum or None

  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  // std:: grad_output = context.fromSsa(this->grad_output());

  // TODO(T51667): Need to handle the cases of other reductions.
  auto expr = pe::Mul(pe::Const(2.0f / input.numElements()),
                      pe::Mul(pe::_3, pe::Sub(pe::_1, pe::_2)));

  poplar::Tensor out = popops::map(context.graph, expr,
                                   {input, target, grad_output}, context.seq);

  context.tensors.insert({result(), out});
}

poplar::Tensor maskTensor(CompilerContext &context, poplar::Tensor &tensor,
                          poplar::Tensor &target, const int ignore_index) {
  auto ignore_index_tensor =
      context.graph.addConstant(target.elementType(), {}, ignore_index);
  context.graph.setTileMapping(ignore_index_tensor, 0);
  auto loss_mask_bool =
      popops::map(context.graph, popops::expr::BinaryOpType::NOT_EQUAL, target,
                  ignore_index_tensor, context.seq);
  auto loss_mask = popops::cast(context.graph, loss_mask_bool,
                                tensor.elementType(), context.seq);
  auto total_elements = popops::reduce(context.graph, loss_mask, {0},
                                       {popops::Operation::ADD}, context.seq);
  if (loss_mask.rank() != tensor.rank()) {
    loss_mask = loss_mask.expand({1});
  }
  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY,
                     tensor, loss_mask, context.seq);
  return total_elements;
}

void nll_loss::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  const int ignore_index = this->ignore_index();
  const int reduction = this->reduction();

  poplar::Tensor labels1d = target.flatten();
  poplar::Tensor probs2d = input.flatten(0, input.rank() - 1);
  poplar::Tensor onehot = context.graph.clone(probs2d);
  popops::encodeOneHot(context.graph, labels1d, onehot, context.seq);

  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY,
                     onehot, probs2d, context.seq);
  poplar::Tensor out = popops::reduce(context.graph, onehot, {1},
                                      {popops::Operation::ADD}, context.seq);
  auto total_elements =
      context.graph.addConstant(input.elementType(), {}, target.shape()[0]);
  context.graph.setTileMapping(total_elements, 0);

  if (ignore_index >= 0) {
    total_elements = maskTensor(context, out, target, ignore_index);
  }
  if (reduction != 0) { // other than none
    out = popops::reduce(context.graph, out, {0}, {popops::Operation::ADD},
                         context.seq);
    if (reduction == 1) { // mean reduce
      popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, out,
                         total_elements, context.seq);
    }
  }
  popops::mapInPlace(context.graph, popops::expr::UnaryOpType::NEGATE, out,
                     context.seq);

  context.tensors.insert({result(), out});
  context.tensors.insert({total_weight(), out});
}

void nll_loss_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  const int ignore_index = this->ignore_index();
  const int reduction = this->reduction();

  poplar::Tensor labels1d = target.flatten();
  poplar::Tensor probs2d = input.flatten(0, input.rank() - 1);
  poplar::Tensor onehot = context.graph.clone(probs2d);
  popops::encodeOneHot(context.graph, labels1d, onehot, context.seq);
  popops::mapInPlace(context.graph, pe::UnaryOpType::NEGATE, onehot,
                     context.seq);
  poplar::Tensor grad = onehot.reshape(input.shape());
  auto total_elements =
      context.graph.addConstant(input.elementType(), {}, target.shape()[0]);
  context.graph.setTileMapping(total_elements, 0);
  if (ignore_index >= 0) {
    total_elements = maskTensor(context, grad, labels1d, ignore_index);
  }

  if (reduction == 1) { // mean reduce
    popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, grad,
                       total_elements, context.seq);
  } else if (reduction == 0) { // none reduction
    grad_output = grad_output.reshape({grad.shape()[0], 1});
  }
  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY, grad,
                     grad_output, context.seq);
  context.tensors.insert({result(), grad});
}

poplar::Tensor bce(CompilerContext &context, poplar::Tensor &pred,
                   poplar::Tensor &target) {
  target = popops::cast(context.graph, target, pred.elementType(), context.seq);
  // Create an epsilon value
  double eps_f = 1.0e-7;
  // if using fp16, increase the eps to avoid underfloat
  if (pred.elementType() == poplar::HALF) {
    eps_f = 6.104e-05;
  }
  poplar::Tensor eps = createConstant(context, pred.elementType(), {},
                                      std::vector<double>{eps_f});
  poplar::Tensor one = context.graph.addConstant(pred.elementType(), {1}, 1.0);
  context.graph.setTileMapping(one, 0);
  // log_prob
  poplar::Tensor log_prob =
      popops::map(context.graph, pe::Log(pe::Max(pe::_1, pe::_2)), {pred, eps},
                  context.seq);
  // log(1-prob)
  poplar::Tensor inverse_log_prob = popops::map(
      context.graph, pe::Log(pe::Max(pe::Sub(pe::_3, pe::_1), pe::_2)),
      {pred, eps, one}, context.seq);
  poplar::Tensor out =
      popops::map(context.graph,
                  pe::Add(pe::Mul(pe::_3, pe::_1),
                          pe::Mul(pe::Sub(pe::_4, pe::_3), pe::_2)),
                  {log_prob, inverse_log_prob, target, one}, context.seq);
  return out;
}

void binary_cross_entropy::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  const int reduction = this->reduction();
  poplar::Tensor out = bce(context, input, target);

  if (reduction != 0) { // other than none
    out = popops::reduce(context.graph, out, {0}, {popops::Operation::ADD},
                         context.seq);
    if (reduction == 1) { // mean reduce
      auto total_elements =
          createConstant(context, input.elementType(), {},
                         std::vector<uint64_t>{target.shape()[0]});
      popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, out,
                         total_elements, context.seq);
    }
  }
  popops::mapInPlace(context.graph, popops::expr::UnaryOpType::NEGATE, out,
                     context.seq);

  context.tensors.insert({result(), out});
}

void binary_cross_entropy_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  const int reduction = this->reduction();
  poplar::Tensor one = createConstant(context, input.elementType(), {1},
                                      std::vector<float>{1.0});

  poplar::Tensor first_term =
      popops::map(context.graph, popops::expr::BinaryOpType::DIVIDE, target,
                  input, context.seq);

  poplar::Tensor second_term =
      popops::map(context.graph,
                  pe::Divide(pe::Sub(pe::_1, pe::_3), pe::Sub(pe::_3, pe::_2)),
                  {target, input, one}, context.seq);

  poplar::Tensor grad =
      popops::map(context.graph, pe::Neg(pe::Add(pe::_1, pe::_2)),
                  {first_term, second_term}, context.seq);
  if (reduction == 1) { // mean reduce
    auto total_elements =
        createConstant(context, input.elementType(), {},
                       std::vector<uint64_t>{target.shape()[0]});
    popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, grad,
                       total_elements, context.seq);
  }

  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY, grad,
                     grad_output, context.seq);
  context.tensors.insert({result(), grad});
}

void binary_cross_entropy_with_logits::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  const int reduction = this->reduction();
  input = popops::sigmoid(context.graph, input, context.seq);
  poplar::Tensor out = bce(context, input, target);

  if (reduction != 0) { // other than none
    out = popops::reduce(context.graph, out, {0}, {popops::Operation::ADD},
                         context.seq);
    if (reduction == 1) { // mean reduce
      auto total_elements =
          createConstant(context, input.elementType(), {},
                         std::vector<uint64_t>{target.shape()[0]});
      popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, out,
                         total_elements, context.seq);
    }
  }
  popops::mapInPlace(context.graph, popops::expr::UnaryOpType::NEGATE, out,
                     context.seq);

  context.tensors.insert({result(), out});
}

void binary_cross_entropy_with_logits_backward::lowerToPoplar(
    CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  const int reduction = this->reduction();
  poplar::Tensor prob = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::SIGMOID, input, context.seq);

  poplar::Tensor grad =
      popops::map(context.graph, popops::expr::BinaryOpType::SUBTRACT, prob,
                  target, context.seq);
  if (reduction == 1) { // mean reduce
    auto total_elements =
        createConstant(context, input.elementType(), {},
                       std::vector<uint64_t>{target.shape()[0]});
    popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, grad,
                       total_elements, context.seq);
  }
  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY, grad,
                     grad_output, context.seq);
  context.tensors.insert({result(), grad});
}

} // namespace poptorch_ir
