// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/CompilerHelpers.hpp"
#include <popops/ElementWise.hpp>

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

} // namespace poptorch_ir
