// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_POPTORCH_STATICGATHER_HPP
#define GUARD_POPTORCH_STATICGATHER_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/opx.hpp>

namespace poptorch_custom_ops {

class FastGatherLastDimOp : public popart::Op {
public:
  FastGatherLastDimOp(popart::OperatorIdentifier const &opid_,
                      popart::Op::Settings const &settings_,
                      std::string const &debug_str);

  FastGatherLastDimOp(const FastGatherLastDimOp &) = default;
  FastGatherLastDimOp &operator=(const FastGatherLastDimOp &) = delete;
  ~FastGatherLastDimOp() override = default;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  int64_t getAxis() const { return _axis; }
  popart::Shape getInShape() const { return _in_shape; }
  popart::Shape getOutShape() const { return _out_shape; }
  std::string const &getDebugStr() const { return _debug_str; }

private:
  int64_t _axis;
  popart::Shape _in_shape;
  popart::Shape _out_shape;
  std::string _debug_str;
};

class FastGatherLastDimOpx : public popart::popx::Opx {
public:
  FastGatherLastDimOpx(popart::Op *, popart::popx::Devicex *);
  ~FastGatherLastDimOpx() override = default;

  void grow(poplar::program::Sequence &prog) const final;

private:
  static poplar::Tensor
  addGraphProg(poplar::Graph &graph, poplar::program::Sequence &prog,
               poplar::Tensor const &data_tensor,
               poplar::Tensor const &idx_tensor,
               std::vector<std::size_t> const &fwd_out_shape);
};

class FastGatherLastDimGradOp : public popart::Op {
public:
  explicit FastGatherLastDimGradOp(const FastGatherLastDimOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  virtual void setup() {
    this->outInfo(0) = {this->inInfo(0).dataType(), _fwd_in_shape};
  }

  /* Describes the relationship of the inputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
    static const std::vector<popart::GradInOutMapper> in_info = {
        // The input of grad op at index 0 is the gradient of the output at
        // index 0 of the non-grad op
        {0, 0, popart::GradOpInType::GradOut},

        // The input of grad op at index 1 is the input at index 1
        // of the non-grad op
        {1, 1, popart::GradOpInType::In}};
    return in_info;
  }

  /* Describes the relationship of the outputs of the grad op to the
     inputs/outputs of the non-grad op */
  virtual const std::map<int, int> &gradOutToNonGradIn() const {
    static const std::map<int, int> out_info = {
        // The output at index 0 is dLhs, i.e the gradient of the input at index
        // 0 of non-grad op
        {0, 0},
    };
    return out_info;
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  const std::string &getDebugStr() const { return _debug_str; }
  popart::Shape getFwdInShape() const { return _fwd_in_shape; }

private:
  int64_t _axis;
  popart::Shape _fwd_in_shape;
  std::string _debug_str;
};

class FastGatherLastDimGradOpx : public popart::popx::Opx {
public:
  FastGatherLastDimGradOpx(popart::Op *op, popart::popx::Devicex *devicex);
  ~FastGatherLastDimGradOpx() override = default;

  void grow(poplar::program::Sequence &prog) const final;

private:
  static poplar::Tensor
  addGraphProg(poplar::Graph &graph, poplar::program::Sequence &prog,
               poplar::Tensor const &grad_output_tensor,
               poplar::Tensor &grad_input_tensor,
               std::vector<std::size_t> const &fwd_in_shape,
               poplar::Tensor const &idx_tensor);
};

} // namespace poptorch_custom_ops

#endif
