// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// This tests the use of many attributes

#include <popart/op.hpp>
#include <popart/opidentifier.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace custom_operators {
const popart::OperatorIdentifier many_attribute = {
    "test.poptorch", "ManyAttributeOp", 1, 1, 1}; // NOLINT
} // namespace custom_operators

class ManyAttributeOp;
class ManyAttributeOpx;

// Adds one if all attributes in the creator were correct, otherwise acts
// as an identity function
class ManyAttributeOp : public popart::Op {
public:
  ManyAttributeOp(const popart::OperatorIdentifier &_opid, bool all_passed,
                  const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _all_passed(all_passed) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<popart::Op> clone() const final {
    return std::unique_ptr<popart::Op>(new ManyAttributeOp(*this));
  }

  bool allPassed() { return _all_passed; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  bool _all_passed;
};

popart::OpCreator<ManyAttributeOp> many_attribute_op(
    {{custom_operators::many_attribute, {}}},
    [](const popart::OpCreatorInfo &info) {
      bool correct = false;

      // Have 2 of each kind of attribute
      if (info.attributes.getAttribute<popart::Attributes::Float>(
              "float_one") == 1.0 &&
          info.attributes.getAttribute<popart::Attributes::Float>(
              "float_minus_two") == -2.0 &&
          info.attributes.getAttribute<popart::Attributes::Int>("int_zero") ==
              0 &&
          info.attributes.getAttribute<popart::Attributes::Int>(
              "int_minus_five") == -5 &&
          info.attributes.getAttribute<popart::Attributes::Floats>(
              "floats_one_two_three") == std::vector<float>{1.0, 2.0, 3.0} &&
          info.attributes.getAttribute<popart::Attributes::Floats>(
              "floats_minus_one_two_three") ==
              std::vector<float>{-1.0, -2.0, -3.0} &&
          info.attributes.getAttribute<popart::Attributes::Ints>(
              "ints_one_two_three") == std::vector<int64_t>{1, 2, 3} &&
          info.attributes.getAttribute<popart::Attributes::Ints>(
              "ints_minus_one_two_three") == std::vector<int64_t>{-1, -2, -3} &&
          info.attributes.getAttribute<popart::Attributes::String>(
              "a_string") == "string with quotes and slash \" ' \\ end" &&
          info.attributes.getAttribute<popart::Attributes::Strings>("strs") ==
              std::vector<std::string>{"\x01", "\x02", "\x03"}) {
        correct = true;
      }

      return std::unique_ptr<popart::Op>(
          new ManyAttributeOp(info.opid, correct, info.settings));
    },
    true);

class ManyAttributeOpx : public popart::popx::Opx {
public:
  ManyAttributeOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<ManyAttributeOp>(op, custom_operators::many_attribute);
    _all_passed = dynamic_cast<ManyAttributeOp *>(op)->allPassed();
  }

  void grow(poplar::program::Sequence &prog) const override {
    auto in_tensor = getInTensor(0);
    auto const_tensor =
        graph().addConstant(in_tensor.elementType(), {1}, 1, "one");
    graph().setTileMapping(const_tensor, 0);

    if (_all_passed) {
      auto out_tensor =
          popops::add(graph(), in_tensor, const_tensor, prog, debugContext());
      setOutTensor(0, out_tensor);
    } else {
      setOutTensor(0, in_tensor);
    }
  }

private:
  bool _all_passed;
};

static popart::popx::OpxCreator<ManyAttributeOpx>
    many_attributes_opx_creator(custom_operators::many_attribute);

} // extern "C"
