// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// This tests the use of the list of strings attribute

#include <popart/op.hpp>
#include <popart/operators.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Reduce.hpp>

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace custom_operators {
const popart::OperatorIdentifier three_reduce = {
    "test.poptorch", "ThreeReduceOp", 1, 3, 3}; // NOLINT
} // namespace custom_operators

class ThreeReduceOp;
class ThreeReduceOpx;

class ThreeReduceOp : public popart::Op {
public:
  ThreeReduceOp(const popart::OperatorIdentifier &_opid,
                std::vector<std::string> reductions,
                const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _reductions(std::move(reductions)) {}

  void setup() override {
    for (unsigned int i = 0; i < 3; i++) {
      auto in_tensor = inInfo(i);
      popart::Shape out_shape({});

      outInfo(i).set(in_tensor.dataType(), out_shape);
    }
  }

  std::unique_ptr<popart::Op> clone() const final {
    return std::unique_ptr<popart::Op>(new ThreeReduceOp(*this));
  }

  const std::vector<std::string> &getReductions() { return _reductions; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const std::vector<std::string> _reductions;
};

popart::OpCreator<ThreeReduceOp> three_reduce_op(
    {{custom_operators::three_reduce, {}}},
    [](const popart::OpCreatorInfo &info) {
      auto reductions =
          info.attributes.getAttribute<popart::Attributes::Strings>(
              "reductions");

      return std::unique_ptr<popart::Op>(
          new ThreeReduceOp(info.opid, reductions, info.settings));
    },
    true);

class ThreeReduceOpx : public popart::popx::Opx {
public:
  ThreeReduceOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<ThreeReduceOp>(op, custom_operators::three_reduce);

    auto reductions = dynamic_cast<ThreeReduceOp *>(op)->getReductions();
    _mean.reserve(reductions.size());

    for (auto &reduction : reductions) {
      if (reduction == "mean") {
        _mean.emplace_back(true);
      } else {
        _mean.emplace_back(false);
        if (reduction != "sum") {
          exit(1);
        }
      }
    }
  }

  void grow(poplar::program::Sequence &prog) const override {
    for (unsigned int input_num = 0; input_num < 3; input_num++) {
      const poplar::Tensor &in_tensor(getInTensor(input_num));
      auto in_tensor_1_d = in_tensor.flatten();

      double scale = 1.0;
      if (_mean[input_num]) {
        scale /= in_tensor_1_d.dim(0);
      }

      auto scale_tensor =
          graph().addConstant(poplar::FLOAT, {}, scale, "scale");
      graph().setTileMapping(scale_tensor, 0);
      auto out_tensor =
          popops::reduce(graph(), in_tensor_1_d, {0},
                         {popops::Operation::ADD, false, scale_tensor}, prog,
                         debugContext("thee_reduce"));

      setOutTensor(input_num, out_tensor);
    }
  }

private:
  // Mean if true, otherwise sum
  std::vector<bool> _mean;
};

static popart::popx::OpxCreator<ThreeReduceOpx>
    reduce_opx_creator(custom_operators::three_reduce);

} // extern "C"
