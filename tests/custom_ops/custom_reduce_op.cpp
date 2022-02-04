// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// This tests the use of the string attribute

#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/operators.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Reduce.hpp>

// Use extern to avoid mangled names when importing to python
extern "C" {

namespace custom_operators {
const popart::OperatorIdentifier reduce = {"test.poptorch", "ReduceOp", 1, 1,
                                           1}; // NOLINT
} // namespace custom_operators

class ReduceOp;
class ReduceOpx;

class ReduceOp : public popart::Op {
public:
  ReduceOp(const popart::OperatorIdentifier &_opid, std::string reduction,
           const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _reduction(std::move(reduction)) {}

  void setup() override {
    auto in_tensor = inInfo(0);
    popart::Shape out_shape({});
    outInfo(0).set(in_tensor.dataType(), out_shape);
  }

  std::unique_ptr<popart::Op> clone() const final {
    return std::unique_ptr<popart::Op>(new ReduceOp(*this));
  }

  std::string getReduction() { return _reduction; }

  // An estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::string _reduction;
};

popart::OpCreator<ReduceOp> reduce_op(
    {{custom_operators::reduce, {}}},
    [](const popart::OpCreatorInfo &info) {
      auto reduction = info.attributes.getAttribute<popart::Attributes::String>(
          "reduction", "mean");

      return std::unique_ptr<popart::Op>(
          new ReduceOp(info.opid, reduction, info.settings));
    },
    true);

class ReduceOpx : public popart::popx::Opx {
public:
  ReduceOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<ReduceOp>(op, custom_operators::reduce);

    if (dynamic_cast<ReduceOp *>(op)->getReduction() == "mean") {
      _mean = true;
    } else if (dynamic_cast<ReduceOp *>(op)->getReduction() == "sum") {
      _mean = false;
    } else {
      exit(1);
    }
  }

  void grow(poplar::program::Sequence &prog) const override {
    const poplar::Tensor &in_tensor(getInTensor(0));
    auto in_tensor_1_d = in_tensor.flatten();

    double scale = 1.0;
    if (_mean) {
      scale /= in_tensor_1_d.dim(0);
    }

    auto scale_tensor = graph().addConstant(poplar::FLOAT, {}, scale, "scale");
    graph().setTileMapping(scale_tensor, 0);
    auto out_tensor =
        popops::reduce(graph(), in_tensor_1_d, {0},
                       {popops::Operation::ADD, false, scale_tensor}, prog,
                       debugContext("reduce"));

    setOutTensor(0, out_tensor);
  }

private:
  // Mean if true, otherwise sum
  bool _mean;
};

static popart::popx::OpxCreator<ReduceOpx>
    reduce_opx_creator(custom_operators::reduce);

} // extern "C"
