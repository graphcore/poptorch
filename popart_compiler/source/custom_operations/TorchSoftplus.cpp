// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "TorchSoftplus.hpp"
#include "CustomOps.hpp"
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/softplusx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>

namespace poptorch_custom_ops {
TorchSoftplusOp::TorchSoftplusOp(const popart::OperatorIdentifier &opid_,
                                 float beta, float threshold,
                                 const popart::Op::Settings &opSettings)
    : popart::ElementWiseUnaryOp(opid_, opSettings), _beta(beta),
      _threshold(threshold) {}

std::unique_ptr<popart::Op> TorchSoftplusOp::clone() const {
  return std::make_unique<TorchSoftplusOp>(*this);
}

std::vector<std::unique_ptr<popart::Op>> TorchSoftplusOp::getGradOps() {
  std::vector<std::unique_ptr<popart::Op>> result;
  result.emplace_back(std::make_unique<TorchSoftplusGradOp>(*this));
  return result;
}

std::vector<std::tuple<popart::OperatorIdentifier, float>>
TorchSoftplusOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{poptorch_custom_ops::torch_softplus_inplace, 10}};
}

std::unique_ptr<popart::Op> TorchSoftplusOp::getInplaceVariant(
    const popart::OperatorIdentifier &operator_id) const {
  if (operator_id == poptorch_custom_ops::torch_softplus_inplace) {
    return std::make_unique<TorchSoftplusInplaceOp>(*this);
  }
  return popart::Op::getInplaceVariant(operator_id);
}

void TorchSoftplusOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  popart::Op::appendOutlineAttributes(os);
  os.appendAttribute("beta", beta());
  os.appendAttribute("threshold", threshold());
}

TorchSoftplusInplaceOp::TorchSoftplusInplaceOp(const TorchSoftplusOp &op)
    : popart::ElementWiseInplaceUnaryOp(
          poptorch_custom_ops::torch_softplus_inplace, op.getSettings()),
      _beta(op.beta()), _threshold(op.threshold()) {}

std::unique_ptr<popart::Op> TorchSoftplusInplaceOp::clone() const {
  return std::make_unique<TorchSoftplusInplaceOp>(*this);
}

void TorchSoftplusInplaceOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  popart::Op::appendOutlineAttributes(os);
  os.appendAttribute("beta", beta());
  os.appendAttribute("threshold", threshold());
}

TorchSoftplusGradOp::TorchSoftplusGradOp(const TorchSoftplusOp &fwd_op)
    : popart::ElementWiseNonLinearUnaryGradOp(
          poptorch_custom_ops::torch_softplus_grad, fwd_op),
      _beta(fwd_op.beta()), _threshold(fwd_op.threshold()) {}

std::unique_ptr<popart::Op> TorchSoftplusGradOp::clone() const {
  return std::make_unique<TorchSoftplusGradOp>(*this);
}

void TorchSoftplusGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  popart::Op::appendOutlineAttributes(os);
  os.appendAttribute("beta", beta());
  os.appendAttribute("threshold", threshold());
}

namespace {
popart::OpDefinition::DataTypes dtypes = {
    popart::DataType::UINT8,   popart::DataType::UINT16,
    popart::DataType::UINT32,  popart::DataType::UINT64,
    popart::DataType::INT8,    popart::DataType::INT16,
    popart::DataType::INT32,   popart::DataType::INT64,
    popart::DataType::FLOAT16, popart::DataType::FLOAT};

popart::OpDefinition
    softplus_def({popart::OpDefinition::Inputs({{"input", dtypes}}),
                  popart::OpDefinition::Outputs({{"output", dtypes}}),
                  popart::OpDefinition::Attributes({{"beta", {"*"}},
                                                    {"threshold", {"*"}}})});

popart::OpCreator<TorchSoftplusOp> softplus_creator(
    popart::OpDefinitions({{poptorch_custom_ops::torch_softplus,
                            softplus_def}}),
    [](const popart::OpCreatorInfo &info) {
      float beta =
          info.attributes.getAttribute<popart::Attributes::Float>("beta", 1.0);
      float threshold = info.attributes.getAttribute<popart::Attributes::Float>(
          "threshold", 1.0);
      return std::unique_ptr<popart::Op>(
          new TorchSoftplusOp(info.opid, beta, threshold, info.settings));
    },
    true);

} // namespace

namespace pe = popops::expr;

template <class T>
std::unique_ptr<popart::popx::EwuComputex> create(popart::Op *op) {
  auto *x = dynamic_cast<T *>(op);
  if (x == nullptr) {
    throw popart::error("Invalid torch softplus operator.");
  }

  return TorchSoftplusComputex::get(x->beta(), x->threshold());
}

TorchSoftplusOpx::TorchSoftplusOpx(popart::Op *op,
                                   popart::popx::Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, create<TorchSoftplusOp>(op)) {
  verifyOp<TorchSoftplusOp>(op, {poptorch_custom_ops::torch_softplus});
}

void TorchSoftplusComputex::inplace(snap::program::Sequence &prog,
                                    snap::Graph &graph,
                                    const snap::Tensor &tensor,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &prefix) const {
  // Torch Softplus definition:
  //  1/beta * log[1 + exp(beta * x)] for beta * x <= threshold
  //                                x for beta * x > threshold
  //
  // To avoid overflow when evaluating the exp, we use the following equivalent
  // formula for softplus:
  // 1/beta * log[1 + exp(-abs(beta * x))] + max(x, 0)
  (void)prefix; // unused input parameter
  using ExprPtr = std::unique_ptr<pe::Expr>;
  std::vector<ExprPtr> exprs;
  exprs.push_back(std::make_unique<pe::PlaceHolder>(pe::_1));

  if (_beta != 1.0f) {
    exprs.push_back(std::make_unique<pe::Mul>(pe::Const(_beta), *exprs.back()));
  }

  auto &bx = *exprs.back();

  // log1p(-exp(|beta * x|))
  exprs.push_back(std::make_unique<pe::Exp>(-pe::Abs(*exprs.back())));
  exprs.push_back(std::make_unique<pe::Log1p>(*exprs.back()));

  if (_beta != 1.0f) {
    exprs.push_back(
        std::make_unique<pe::Divide>(*exprs.back(), pe::Const(_beta)));
  }

  // 1/beta * log1p(-exp(|beta * x|)) + max(x, 0)
  exprs.push_back(std::make_unique<pe::Add>(*exprs.back(),
                                            pe::Max(pe::_1, pe::Const(0.0f))));

  // beta * x <= threshold ? 1/beta * log1p(-exp(|beta * x|)) + max(x, 0) : x
  exprs.push_back(std::make_unique<pe::Select>(*exprs.back(), pe::_1,
                                               bx <= pe::Const(_threshold)));

  popops::mapInPlace(graph.getPoplarGraph(), *exprs.back(),
                     {tensor.getPoplarTensor()}, prog.getPoplarSequence(),
                     {dnai, "torch_softplus"});
}

std::unique_ptr<popart::popx::EwuComputex>
TorchSoftplusComputex::get(float beta, float threshold) {
  return std::make_unique<TorchSoftplusComputex>(beta, threshold);
}

TorchSoftplusInplaceOpx::TorchSoftplusInplaceOpx(popart::Op *op,
                                                 popart::popx::Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex,
                                 create<TorchSoftplusInplaceOp>(op)) {
  verifyOp<TorchSoftplusInplaceOp>(op,
                                   poptorch_custom_ops::torch_softplus_inplace);
}

TorchSoftplusGradOpx::TorchSoftplusGradOpx(popart::Op *op,
                                           popart::popx::Devicex *devicex)
    : PopOpx(op, devicex), _beta(), _threshold() {
  verifyOp<TorchSoftplusGradOp>(op, poptorch_custom_ops::torch_softplus_grad);
  auto &grad_op = getOp<TorchSoftplusGradOp>();
  _beta = grad_op.beta();
  _threshold = grad_op.threshold();
}

void TorchSoftplusGradOpx::grow(snap::program::Sequence &prog) const {
  // The derivative of the softplus activation function is:
  //
  // exp(beta*x)/(exp(beta*x) + 1) = 1/(exp(-beta*x) + 1) = sigmoid(beta*x)
  //
  // To match the Torch definition:
  //
  // grad_out = grad_in * sigmoid(beta*x) for beta * x <= threshold
  //            grad_in                   for beta * x > threshold
  const auto &grad_in = getInTensor(TorchSoftplusGradOp::getGradInIndex());
  const auto &fwd_input = getInTensor(TorchSoftplusGradOp::getFwdArgInIndex());

  using ExprPtr = std::unique_ptr<pe::Expr>;
  std::vector<ExprPtr> exprs;
  exprs.push_back(std::make_unique<pe::PlaceHolder>(pe::_2));

  if (_beta != 1.0f) {
    exprs.push_back(std::make_unique<pe::Mul>(pe::Const(_beta), *exprs.back()));
  }

  auto &bx = *exprs.back();

  // grad_in * sigmoid(beta*x)
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, pe::Sigmoid(bx)));

  // beta * x <= threshold ? grad_in * sigmoid(beta*x) : grad_in
  exprs.push_back(std::make_unique<pe::Select>(*exprs.back(), pe::_1,
                                               bx <= pe::Const(_threshold)));

  auto output = popops::map(
      graph().getPoplarGraph(), *exprs.back(),
      {grad_in.getPoplarTensor(), fwd_input.getPoplarTensor()},
      prog.getPoplarSequence(), debugContext("torch_softplus_grad"));

  setOutTensor(TorchSoftplusGradOp::getOutIndex(),
               snap::Tensor{output, graph()});
}

namespace {
popart::popx::OpxCreator<TorchSoftplusOpx>
    softplus_opx(poptorch_custom_ops::torch_softplus);
popart::popx::OpxCreator<TorchSoftplusInplaceOpx>
    softplus_inplace_opx(poptorch_custom_ops::torch_softplus_inplace);
popart::popx::OpxCreator<TorchSoftplusGradOpx>
    softplus_grad_opx(poptorch_custom_ops::torch_softplus_grad);
} // namespace

} // namespace poptorch_custom_ops
