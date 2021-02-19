// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// This tests the use of the int_64/float attributes

#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/opidentifier.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace custom_operators {
const popart::OperatorIdentifier add_vec_scalar_mul_float = {
    "test.poptorch", "AddVecScalarMulFloat", 1, 1, 1}; // NOLINT
} // namespace custom_operators

class AddVecScalarMulFloatOp;
class AddVecScalarMulFloatOpx;

// Add the vec and multiply by the scalar
class AddVecScalarMulFloatOp : public popart::Op {
public:
  AddVecScalarMulFloatOp(const popart::OperatorIdentifier &_opid, float scalar,
                         std::vector<float> vec,
                         const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _scalar(scalar), _vec(std::move(vec)) {}

  void setup() override {
    if (inInfo(0).shape().size() != 1) {
      exit(1);
    }
    if (static_cast<size_t>(inInfo(0).shape()[0]) != _vec.size()) {
      exit(1);
    }
    outInfo(0) = inInfo(0);
  }

  std::unique_ptr<popart::Op> clone() const final {
    return std::unique_ptr<popart::Op>(new AddVecScalarMulFloatOp(*this));
  }

  float getScalar() { return _scalar; }

  const std::vector<float> &getVec() { return _vec; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  float _scalar;
  std::vector<float> _vec;
};

popart::OpCreator<AddVecScalarMulFloatOp> add_vec_scalar_mul_float_op(
    {{custom_operators::add_vec_scalar_mul_float, {}}},
    [](const popart::OpCreatorInfo &info) {
      float scalar = info.attributes.getAttribute<popart::Attributes::Float>(
          "scalar", 0.0f);
      std::vector<float> vec =
          info.attributes.getAttribute<popart::Attributes::Floats>("vec");

      return std::unique_ptr<popart::Op>(
          new AddVecScalarMulFloatOp(info.opid, scalar, vec, info.settings));
    },
    true);

class AddVecScalarMulFloatOpx : public popart::popx::Opx {
public:
  AddVecScalarMulFloatOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<AddVecScalarMulFloatOp>(
        op, custom_operators::add_vec_scalar_mul_float);
    _scalar = dynamic_cast<AddVecScalarMulFloatOp *>(op)->getScalar();
    _vec = dynamic_cast<AddVecScalarMulFloatOp *>(op)->getVec();
  }

  void grow(poplar::program::Sequence &prog) const override {
    auto in_tensor = getInTensor(0);

    auto vec_tensor = graph().addConstant(
        poplar::FLOAT, {_vec.size()},
        poplar::ArrayRef<float>(_vec.data(), _vec.size()), "vec");
    graph().setTileMapping(vec_tensor, 0);

    auto added_tensor =
        popops::add(graph(), in_tensor, vec_tensor, prog, debugPrefix());

    auto scalar_tensor =
        graph().addConstant(poplar::FLOAT, {1}, _scalar, "scale_factor");
    graph().setTileMapping(scalar_tensor, 0);

    auto out_tensor =
        popops::mul(graph(), added_tensor, scalar_tensor, prog, debugPrefix());
    setOutTensor(0, out_tensor);
  }

private:
  float _scalar;
  std::vector<float> _vec;
};

static popart::popx::OpxCreator<AddVecScalarMulFloatOpx>
    add_scalar_float_opx_creator(custom_operators::add_vec_scalar_mul_float);

} // extern "C"
