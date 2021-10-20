// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// This file is based on the example in the PopART User Guide:
// https://docs.sourcevertex.net/files/popart-popart-user-guide-latest/custom_ops.html

#include <memory>

#include <popart/op.hpp>

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace {

// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> makeUnique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace custom_operators {
const popart::OperatorIdentifier leaky_relu = {
    "com.acme", "LeakyRelu", 1, {1, 1}}; // NOLINT
} // namespace custom_operators

namespace custom_grad_operators {
const static popart::OperatorIdentifier LeakyReluGrad = { // NOLINT
    "com.acme",
    "LeakyReluGrad",
    1,
    {1, 1}};
} // namespace custom_grad_operators

class LeakyReluGradOp;

class LeakyReluOp : public popart::Op {
public:
  LeakyReluOp(const popart::OperatorIdentifier &_opid, float alpha_,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _alpha(alpha_) {}

  std::unique_ptr<Op> clone() const final {
    return makeUnique<LeakyReluOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(makeUnique<LeakyReluGradOp>(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

  // Attributes
  float getAlpha() const { return _alpha; }

private:
  float _alpha;
};

static popart::OpDefinition::DataTypes t = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};

static popart::OpDefinition
    leaky_relu_op_def({popart::OpDefinition::Inputs({{"input", t}}),
                       popart::OpDefinition::Outputs({{"output", t}}),
                       popart::OpDefinition::Attributes({{"alpha", {"*"}}})});

static popart::OpCreator<LeakyReluOp> leaky_relu_op_creator(
    popart::OpDefinitions({{custom_operators::leaky_relu, leaky_relu_op_def}}),
    [](const popart::OpCreatorInfo &info) {
      float alpha = info.attributes.getAttribute<popart::Attributes::Float>(
          "alpha", 1e-2f);

      // default epsilon is 10**(-2)
      return makeUnique<LeakyReluOp>(info.opid, alpha, info.settings);
    },
    true);

class LeakyReluOpx : public popart::popx::Opx {
public:
  LeakyReluOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluOp>(op, {custom_operators::leaky_relu});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op = getOp<LeakyReluOp>();

    poplar::Tensor input = getInTensor(0);

    float alpha = op.getAlpha();

    // x < 0.0f ? alpha * x : x
    auto expression = popops::expr::Select(
        popops::expr::Mul(popops::expr::Const(alpha), popops::expr::_1),
        popops::expr::_1,
        popops::expr::Lt(popops::expr::_1, popops::expr::Const(0.0f)));

    popops::mapInPlace(graph(), expression, {input}, prog,
                       debugContext("LeakyRelu"), poplar::OptionFlags());

    setOutTensor(0, input);
  }
};

static popart::popx::OpxCreator<LeakyReluOpx>
    add_scalar_float_opx_creator(custom_operators::leaky_relu);

class LeakyReluGradOp : public popart::Op {
public:
  explicit LeakyReluGradOp(const LeakyReluOp &fwdOp)
      : popart::Op(custom_grad_operators::LeakyReluGrad, fwdOp.settings),
        _alpha(fwdOp.getAlpha()) {}

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<LeakyReluGradOp>(*this);
  }
  void setup() final { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<popart::GradInOutMapper> in_info = {
        {0, 0, popart::GradOpInType::GradOut},
        {1, 0, popart::GradOpInType::In}};
    return in_info;
  }

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> out_info = {{0, 0}};
    return out_info;
  }

  bool requiresRandomSeed() const override { return false; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  float getAlpha() const { return _alpha; }

  // Implementation defined below
  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  // Implementation defined below
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

private:
  float _alpha;
};

class LeakyReluGradOpx : public popart::popx::Opx {
public:
  LeakyReluGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluGradOp>(op, {custom_grad_operators::LeakyReluGrad});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op = getOp<LeakyReluGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    float alpha = op.getAlpha();

    // (grad * (x < 0.0f ? alpha : 1))
    auto expression = popops::expr::Mul(
        popops::expr::Select(
            popops::expr::Const(alpha), popops::expr::Const(1.0f),
            popops::expr::Lt(popops::expr::_2, popops::expr::Const(0.0f))),
        popops::expr::_1);

    auto output =
        popops::map(graph(), expression, {grad, input}, prog,
                    debugContext("LeakyReluGrad"), poplar::OptionFlags());

    setOutTensor(0, output);
  }
};

static popart::popx::OpxCreator<LeakyReluOpx>
    leaky_relu_opx_creator({custom_operators::leaky_relu});
static popart::popx::OpxCreator<LeakyReluGradOpx>
    leaky_relu_grad_opx_creator(custom_grad_operators::LeakyReluGrad);

} // extern "C"
