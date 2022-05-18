
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/CompilerHelpers.hpp"
#include "pytorch_bridge/PytorchBridgeUtils.hpp"
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>

#include "poptorch_logging/Logging.hpp"

namespace pe = popops::expr;

namespace poptorch_ir {

void mse_loss::lowerToPoplar(CompilerContext &context) {
  // aten::mse_loss(Tensor self, Tensor target, int reduction) -> Tensor

  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  const TorchReduction reduction = getTorchReduction(this->reduction());

  auto error_expr = pe::Sub(pe::_1, pe::_2);
  poplar::Tensor sqr_error =
      popops::map(context.graph, pe::Mul(error_expr, error_expr),
                  {input, target}, context.seq);

  if (reduction != TorchReduction::NONE) {
    std::vector<std::size_t> dims(input.rank());
    std::iota(dims.begin(), dims.end(), 0);
    sqr_error = popops::reduce(context.graph, sqr_error, dims,
                               {popops::Operation::ADD}, context.seq);

    if (reduction == TorchReduction::MEAN) {
      sqr_error = popops::map(
          context.graph, pe::Divide(pe::_1, pe::Const(input.numElements())),
          {sqr_error}, context.seq);
    }
  }

  context.addTensor(result(), sqr_error);
}

void mse_loss_backward::lowerToPoplar(CompilerContext &context) {
  // aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int
  // reduction) -> (Tensor)
  //    alpha * (input - target) * grad_output;
  // Where alpha is
  //      2            if Reduction is Sum or None
  //      2 / numel()  if Reduction is Mean

  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  const TorchReduction reduction = getTorchReduction(this->reduction());

  float alpha =
      reduction == TorchReduction::MEAN ? 2.0f / input.numElements() : 2.0f;
  auto expr =
      pe::Mul(pe::Const(alpha), pe::Mul(pe::_3, pe::Sub(pe::_1, pe::_2)));

  poplar::Tensor out = popops::map(context.graph, expr,
                                   {input, target, grad_output}, context.seq);

  context.addTensor(result(), out);
}

poplar::Tensor maskTensor(CompilerContext &context, poplar::Tensor &tensor,
                          poplar::Tensor &target, const int ignore_index) {
  poplar::Tensor ignore_index_tensor =
      createConstant(context, target.elementType(), {}, ignore_index);

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
  const TorchReduction reduction = getTorchReduction(this->reduction());

  poplar::Tensor labels1d = target.flatten();
  poplar::Tensor probs2d = input.flatten(0, input.rank() - 1);
  poplar::Tensor onehot = context.graph.clone(probs2d);
  popops::encodeOneHot(context.graph, labels1d, onehot, context.seq);

  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY,
                     onehot, probs2d, context.seq);
  poplar::Tensor out = popops::reduce(context.graph, onehot, {1},
                                      {popops::Operation::ADD}, context.seq);

  poplar::Tensor total_elements;
  if (target.shape().empty()) {
    total_elements = createConstant(context, input.elementType(), {}, 1);
  } else {
    total_elements =
        createConstant(context, input.elementType(), {}, target.shape()[0]);
  }

  if (ignore_index >= 0) {
    total_elements = maskTensor(context, out, target, ignore_index);
  }
  if (reduction != TorchReduction::NONE) {
    out = popops::reduce(context.graph, out, {0}, {popops::Operation::ADD},
                         context.seq);
    if (reduction == TorchReduction::MEAN) {
      if (ignore_index >= 0) {
        // Prevent dividing by zero
        popops::mapInPlace(
            context.graph, popops::expr::BinaryOpType::MAXIMUM, total_elements,
            createConstant(context, input.elementType(), {}, 1), context.seq);
      }

      popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, out,
                         total_elements, context.seq);
    }
  }
  popops::mapInPlace(context.graph, popops::expr::UnaryOpType::NEGATE, out,
                     context.seq);

  // MLIR might be [] and Poplar [1] so for consistency reshape the Poplar
  // tensor to match the MLIR one.
  out = reshapeToMlirShape(out, this->result().getType());

  context.addTensor(result(), out);
  context.addTensor(total_weight(), total_elements);
}

void nll_loss_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  const int ignore_index = this->ignore_index();
  const TorchReduction reduction = getTorchReduction(this->reduction());

  poplar::Tensor labels1d = target.flatten();
  poplar::Tensor probs2d = input.flatten(0, input.rank() - 1);
  poplar::Tensor onehot = context.graph.clone(probs2d);
  popops::encodeOneHot(context.graph, labels1d, onehot, context.seq);
  popops::mapInPlace(context.graph, pe::UnaryOpType::NEGATE, onehot,
                     context.seq);
  poplar::Tensor grad = onehot.reshape(input.shape());

  poplar::Tensor total_elements;
  if (target.shape().empty()) {
    total_elements = createConstant(context, input.elementType(), {}, 1);
  } else {
    total_elements =
        createConstant(context, input.elementType(), {}, target.shape()[0]);
  }

  if (ignore_index >= 0) {
    total_elements = maskTensor(context, grad, labels1d, ignore_index);
  }

  if (reduction == TorchReduction::MEAN) {
    if (ignore_index >= 0) {
      // Prevent dividing by zero
      popops::mapInPlace(
          context.graph, popops::expr::BinaryOpType::MAXIMUM, total_elements,
          createConstant(context, input.elementType(), {}, 1), context.seq);
    }
    popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, grad,
                       total_elements, context.seq);
  } else if (reduction == TorchReduction::NONE) {
    grad_output = grad_output.reshape({grad.shape()[0], 1});
  }
  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY, grad,
                     grad_output, context.seq);
  context.addTensor(result(), grad);
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
  poplar::Tensor eps = createConstant(context, pred.elementType(), {}, eps_f);
  poplar::Tensor one = createConstant(context, pred.elementType(), {}, 1.0f);

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
  const TorchReduction reduction = getTorchReduction(this->reduction());
  poplar::Tensor out = bce(context, input, target);

  if (this->weight()) {
    popops::mulInPlace(context.graph, out, context.fromSsa(this->weight()),
                       context.seq);
  }

  if (reduction != TorchReduction::NONE) {
    std::vector<size_t> dims(out.rank());
    std::iota(dims.begin(), dims.end(), 0);

    out = popops::reduce(context.graph, out, dims, {popops::Operation::ADD},
                         context.seq);

    if (reduction == TorchReduction::MEAN) {
      poplar::Tensor total_elements =
          createConstant(context, input.elementType(), {}, input.numElements());
      popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, out,
                         total_elements, context.seq);
    }
  }
  popops::mapInPlace(context.graph, popops::expr::UnaryOpType::NEGATE, out,
                     context.seq);

  context.addTensor(result(), out);
}

void binary_cross_entropy_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  const TorchReduction reduction = getTorchReduction(this->reduction());
  poplar::Tensor one = createConstant(context, input.elementType(), {1}, 1.0f);

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

  if (this->weight()) {
    popops::mulInPlace(context.graph, grad, context.fromSsa(this->weight()),
                       context.seq);
  }

  if (reduction == TorchReduction::MEAN) {
    auto total_elements =
        createConstant(context, input.elementType(), {}, input.numElements());
    popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, grad,
                       total_elements, context.seq);
  }

  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY, grad,
                     grad_output, context.seq);
  context.addTensor(result(), grad);
}

void binary_cross_entropy_with_logits::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  const TorchReduction reduction = getTorchReduction(this->reduction());
  input = popops::sigmoid(context.graph, input, context.seq);
  poplar::Tensor out = bce(context, input, target);

  if (reduction != TorchReduction::NONE) {
    out = popops::reduce(context.graph, out, {0}, {popops::Operation::ADD},
                         context.seq);
    if (reduction == TorchReduction::MEAN) {
      auto total_elements =
          createConstant(context, input.elementType(), {}, target.shape()[0]);
      popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, out,
                         total_elements, context.seq);
    }
  }
  popops::mapInPlace(context.graph, popops::expr::UnaryOpType::NEGATE, out,
                     context.seq);

  context.addTensor(result(), out);
}

void binary_cross_entropy_with_logits_backward::lowerToPoplar(
    CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->self());
  poplar::Tensor target = context.fromSsa(this->target());
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  const TorchReduction reduction = getTorchReduction(this->reduction());
  poplar::Tensor prob = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::SIGMOID, input, context.seq);

  poplar::Tensor grad =
      popops::map(context.graph, popops::expr::BinaryOpType::SUBTRACT, prob,
                  target, context.seq);
  if (reduction == TorchReduction::MEAN) {
    auto total_elements =
        createConstant(context, input.elementType(), {}, target.shape()[0]);
    popops::mapInPlace(context.graph, popops::expr::BinaryOpType::DIVIDE, grad,
                       total_elements, context.seq);
  }
  popops::mapInPlace(context.graph, popops::expr::BinaryOpType::MULTIPLY, grad,
                     grad_output, context.seq);
  context.addTensor(result(), grad);
}

} // namespace poptorch_ir
