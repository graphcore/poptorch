// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// This tests the use of the int_64/float attributes

#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/operators.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace custom_operators {
const popart::OperatorIdentifier add_scalar_float = {
    "test.poptorch", "AddScalarFloat", 1, 1, 1}; // NOLINT
} // namespace custom_operators

class AddScalarFloatOp;
class AddScalarFloatOpx;

class AddScalarFloatOp : public popart::Op {
public:
  AddScalarFloatOp(const popart::OperatorIdentifier &_opid, float scalar,
                   const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _scalar(scalar) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<popart::Op> clone() const final {
    return std::unique_ptr<popart::Op>(new AddScalarFloatOp(*this));
  }

  float getScalar() { return _scalar; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  float _scalar;
};

popart::OpCreator<AddScalarFloatOp> add_scalar_float_op(
    {{custom_operators::add_scalar_float, {}}},
    [](const popart::OpCreatorInfo &info) {
      float scalar = info.attributes.getAttribute<popart::Attributes::Float>(
          "scalar", 0.0f);

      return std::unique_ptr<popart::Op>(
          new AddScalarFloatOp(info.opid, scalar, info.settings));
    },
    true);

class AddScalarFloatOpx : public popart::popx::Opx {
public:
  AddScalarFloatOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<AddScalarFloatOp>(op, custom_operators::add_scalar_float);
    _scalar = dynamic_cast<AddScalarFloatOp *>(op)->getScalar();
  }

  void grow(poplar::program::Sequence &prog) const override {
    auto in_tensor = getInTensor(0);
    auto const_tensor = graph().addConstant(in_tensor.elementType(), {1},
                                            _scalar, "scale_factor");
    graph().setTileMapping(const_tensor, 0);

    auto out_tensor =
        popops::add(graph(), in_tensor, const_tensor, prog, debugContext());
    setOutTensor(0, out_tensor);
  }

private:
  float _scalar;
};

static popart::popx::OpxCreator<AddScalarFloatOpx>
    add_scalar_float_opx_creator(custom_operators::add_scalar_float);

namespace custom_operators {
const popart::OperatorIdentifier add_scalar_int = {
    "test.poptorch", "AddScalarInt", 1, 1, 1}; // NOLINT
} // namespace custom_operators

class AddScalarIntOp;
class AddScalarIntOpx;

class AddScalarIntOp : public popart::Op {
public:
  AddScalarIntOp(const popart::OperatorIdentifier &_opid, std::int64_t scalar,
                 const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _scalar(scalar) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<popart::Op> clone() const final {
    return std::unique_ptr<popart::Op>(new AddScalarIntOp(*this));
  }

  std::int64_t getScalar() { return _scalar; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::int64_t _scalar;
};

popart::OpCreator<AddScalarIntOp> add_scalar_int_op(
    {{custom_operators::add_scalar_int, {}}},
    [](const popart::OpCreatorInfo &info) {
      auto scalar =
          info.attributes.getAttribute<popart::Attributes::Int>("scalar", 0);

      return std::unique_ptr<popart::Op>(
          new AddScalarIntOp(info.opid, scalar, info.settings));
    },
    true);

class AddScalarIntOpx : public popart::popx::Opx {
public:
  AddScalarIntOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<AddScalarIntOp>(op, custom_operators::add_scalar_int);
    _scalar = dynamic_cast<AddScalarIntOp *>(op)->getScalar();
  }

  void grow(poplar::program::Sequence &prog) const override {
    auto in_tensor = getInTensor(0);
    auto const_tensor = graph().addConstant(in_tensor.elementType(), {1},
                                            _scalar, "scale_factor");
    graph().setTileMapping(const_tensor, 0);

    auto out_tensor =
        popops::add(graph(), in_tensor, const_tensor, prog, debugContext());
    setOutTensor(0, out_tensor);
  }

private:
  int64_t _scalar;
};

static popart::popx::OpxCreator<AddScalarIntOpx>
    add_scalar_int_opx_creator(custom_operators::add_scalar_int);

} // extern "C"
