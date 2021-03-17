// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// This tests the use of the int_64/float list attributes

#include <vector>

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
const popart::OperatorIdentifier add_scalar_vec_float = {
    "test.poptorch", "AddScalarVecFloat", 1, 1, 1}; // NOLINT
} // namespace custom_operators

class AddScalarVecFloatOp;
class AddScalarVecFloatOpx;

class AddScalarVecFloatOp : public popart::Op {
public:
  AddScalarVecFloatOp(const popart::OperatorIdentifier &_opid,
                      std::vector<float> vec,
                      const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _vec(std::move(vec)) {}

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
    return std::unique_ptr<popart::Op>(new AddScalarVecFloatOp(*this));
  }

  const std::vector<float> &getVec() { return _vec; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::vector<float> _vec;
};

popart::OpCreator<AddScalarVecFloatOp> add_scalar_vec_float_op(
    {{custom_operators::add_scalar_vec_float, {}}},
    [](const popart::OpCreatorInfo &info) {
      std::vector<float> vec =
          info.attributes.getAttribute<popart::Attributes::Floats>("vec");

      return std::unique_ptr<popart::Op>(
          new AddScalarVecFloatOp(info.opid, vec, info.settings));
    },
    true);

class AddScalarVecFloatOpx : public popart::popx::Opx {
public:
  AddScalarVecFloatOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<AddScalarVecFloatOp>(op, custom_operators::add_scalar_vec_float);
    _vec = dynamic_cast<AddScalarVecFloatOp *>(op)->getVec();
  }

  void grow(poplar::program::Sequence &prog) const override {
    auto in_tensor = getInTensor(0);
    auto const_tensor = graph().addConstant(
        poplar::FLOAT, {_vec.size()},
        poplar::ArrayRef<float>(_vec.data(), _vec.size()), "vec");
    graph().setTileMapping(const_tensor, 0);

    auto out_tensor =
        popops::add(graph(), in_tensor, const_tensor, prog, debugContext());
    setOutTensor(0, out_tensor);
  }

private:
  std::vector<float> _vec;
};

static popart::popx::OpxCreator<AddScalarVecFloatOpx>
    add_scalar_vec_float_opx_creator(custom_operators::add_scalar_vec_float);

namespace custom_operators {
const popart::OperatorIdentifier add_scalar_vec_int = {
    "test.poptorch", "AddScalarVecInt", 1, 1, 1}; // NOLINT
} // namespace custom_operators

class AddScalarVecIntOp;
class AddScalarVecIntOpx;

class AddScalarVecIntOp : public popart::Op {
public:
  AddScalarVecIntOp(const popart::OperatorIdentifier &_opid,
                    std::vector<int64_t> vec,
                    const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _vec(std::move(vec)) {}

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
    return std::unique_ptr<popart::Op>(new AddScalarVecIntOp(*this));
  }

  const std::vector<int64_t> &getVec() { return _vec; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::vector<int64_t> _vec;
};

popart::OpCreator<AddScalarVecIntOp> add_scalar_vec_int_op(
    {{custom_operators::add_scalar_vec_int, {}}},
    [](const popart::OpCreatorInfo &info) {
      std::vector<int64_t> vec =
          info.attributes.getAttribute<popart::Attributes::Ints>("vec");

      return std::unique_ptr<popart::Op>(
          new AddScalarVecIntOp(info.opid, vec, info.settings));
    },
    true);

class AddScalarVecIntOpx : public popart::popx::Opx {
public:
  AddScalarVecIntOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<AddScalarVecIntOp>(op, custom_operators::add_scalar_vec_int);
    _vec = dynamic_cast<AddScalarVecIntOp *>(op)->getVec();
  }

  void grow(poplar::program::Sequence &prog) const override {
    auto in_tensor = getInTensor(0);
    auto const_tensor = graph().addConstant(
        poplar::INT, {_vec.size()},
        poplar::ArrayRef<int64_t>(_vec.data(), _vec.size()), "vec");
    graph().setTileMapping(const_tensor, 0);

    auto out_tensor =
        popops::add(graph(), in_tensor, const_tensor, prog, debugContext());
    setOutTensor(0, out_tensor);
  }

private:
  std::vector<int64_t> _vec;
};

static popart::popx::OpxCreator<AddScalarVecIntOpx>
    add_scalar_vec_int_opx_creator(custom_operators::add_scalar_vec_int);

} // extern "C"
