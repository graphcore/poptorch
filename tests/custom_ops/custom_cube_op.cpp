// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// This file has been lifted directly from the PopART examples. See file there
// for usage. Modified to take in and return two tensors.

#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>

#include <popart/ir.hpp>

#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <poprand/RandomGen.hpp>

#include <popops/ElementWise.hpp>

#include <popart/names.hpp>
#include <popart/opidentifier.hpp>

namespace {

// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace CustomOperators {
const popart::OperatorIdentifier Cube = {
    "com.acme", "Cube", 1, {2, 2}}; // NOLINT
} // namespace CustomOperators
namespace CustomGradOperators {
const static popart::OperatorIdentifier CubeGrad = { // NOLINT
    "com.acme",
    "CubeGrad",
    1,
    {2, 2}};
} // namespace CustomGradOperators

// For training with a custom Op, four classes need to be implemented,
// one for each of:
// {forward, gradient} x {Op, Opx}.
//
// If only inference is required, then two classes need to be implemented:
// {forward} x {Op, Opx}.
//
// The Op is a poplar/hardware agnostic description of the computation.
// the Opx is the poplar implementation of the Op.
//
// We do training in this example, so the four classes implemented are:
//
class CubeOp;
class CubeGradOp;
class CubeOpx;
class CubeGradOpx;

// The forward Op

class CubeOp : public popart::Op {
public:
  CubeOp(const popart::OperatorIdentifier &_opid,
         const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  // Configure the output popart Tensor
  void setup() override {
    outInfo(0) = inInfo(0);
    outInfo(1) = inInfo(1);
  }

  std::unique_ptr<Op> clone() const final { return make_unique<CubeOp>(*this); }
  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(make_unique<CubeGradOp>(*this));
    return upops;
  }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpCreator<CubeOp> cubeOpCreator({{CustomOperators::Cube, {}}},
                                               true);

// The forward Opx (poplar implementation of the forward Op)

class CubeOpx : public popart::popx::Opx {
public:
  CubeOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // Not strictly necessary, we check that op is castable to a CubeOp *.
    verifyOp<CubeOp>(op, CustomOperators::Cube);
  }

  void grow(poplar::program::Sequence &prog) const override {
    auto output = popops::map(
        graph(),
        popops::expr::Add(popops::expr::Mul(popops::expr::Mul(popops::expr::_1,
                                                              popops::expr::_1),
                                            popops::expr::_1),
                          popops::expr::_2),
        {getInTensor(0), getInTensor(1)}, prog, debugPrefix());

    setOutTensor(0, output);

    auto output2 = popops::map(
        graph(),
        popops::expr::Mul(popops::expr::Mul(popops::expr::_1, popops::expr::_1),
                          popops::expr::_1),
        {getInTensor(0)}, prog, debugPrefix());

    setOutTensor(1, output2);
  }
};

// The gradient Op
class CubeGradOp : public popart::Op {
public:
  explicit CubeGradOp(const popart::Op &fwdOp)
      : popart::Op(CustomGradOperators::CubeGrad, fwdOp.getSettings()) {}

  std::unique_ptr<Op> clone() const final {
    return make_unique<CubeGradOp>(*this);
  }

  // same comment as for CubeOp, for running shape/type inference "statically"
  void setup() override { outInfo(0) = inInfo(0); }

  // function describing the inputs and output(s) of CubeGradOp
  // The Gradient Op which we are implementing (CubeGradOp) has 2 inputs.
  // The input at index 0 is:
  // the gradient of the 0'th output Tensor of the CubeOp.
  // The input at index 1 is :
  // the 0'th output Tensor of the CubeOp.
  // Supposing the CubeOp has input Tensor T0 and output Tensor T1,
  //
  //  input at index 0 (T0)
  //         |
  //       CubeOp
  //         |
  //  output at index 0 (T1)
  //
  // Then the picture described by the map below looks like,
  //
  //
  //   input at index 0 (gradient of T1)
  //        |   input at index 1 (T1)
  //        |     |
  //        |     |
  //       CubeGradOp
  //           |
  //           |
  //  output at index 0 (gradient of T0)
  //
  const std::vector<popart::GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<popart::GradInOutMapper> inInfo = {
        {0, 0, popart::GradOpInType::GradOut},
        {1, 1, popart::GradOpInType::Out}};
    return inInfo;
  }
  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class CubeGradOpx : public popart::popx::Opx {
public:
  CubeGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<CubeGradOp>(op, CustomGradOperators::CubeGrad);
  }

  // Create the gradient poplar::Tensor, which is
  // 3 * input_to_cube**2 * gradient_of_cube_output
  void grow(poplar::program::Sequence &prog) const final {
    insert(
        outId(0),
        popops::map(graph(),
                    popops::expr::Mul(
                        popops::expr::Const(3),
                        popops::expr::Mul(popops::expr::Mul(popops::expr::_1,
                                                            popops::expr::_1),
                                          popops::expr::_2)),
                    {getInTensor(0), getInTensor(1)}, // FwdOut, GradOut
                    prog, debugPrefix()));
  }
};

static popart::popx::OpxCreator<CubeOpx> cubeOpxCreator(CustomOperators::Cube);
static popart::popx::OpxCreator<CubeGradOpx>
    cubeGradOpxCreator(CustomGradOperators::CubeGrad);
}

static popart::RegisterShapeInferenceFunction
    cubeOpShapeInference(CustomOperators::Cube,
                         [](auto &ctx) { ctx.outInfo(0) = ctx.inInfo(0); });
