// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_POPTORCH_SOFTPLUS_HPP
#define GUARD_POPTORCH_SOFTPLUS_HPP
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace poptorch_custom_ops {

class TorchSoftplusOp : public popart::ElementWiseUnaryOp {
public:
  TorchSoftplusOp(const popart::OperatorIdentifier &opid, float beta,
                  float threshold, const popart::Op::Settings &settings);

  std::unique_ptr<popart::Op> clone() const final;
  std::vector<std::unique_ptr<popart::Op>> getGradOps() final;

  std::vector<std::tuple<popart::OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<popart::Op>
  getInplaceVariant(const popart::OperatorIdentifier &id) const final;

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const final;

  float beta() const { return _beta; }
  float threshold() const { return _threshold; }

private:
  float _beta;
  float _threshold;
};

class TorchSoftplusInplaceOp : public popart::ElementWiseInplaceUnaryOp {
public:
  explicit TorchSoftplusInplaceOp(const TorchSoftplusOp &op);
  std::unique_ptr<popart::Op> clone() const final;

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const final;

  float beta() const { return _beta; }
  float threshold() const { return _threshold; }

private:
  float _beta;
  float _threshold;
};

class TorchSoftplusGradOp : public popart::ElementWiseNonLinearUnaryGradOp {
public:
  explicit TorchSoftplusGradOp(const TorchSoftplusOp &fwd_op);
  std::unique_ptr<popart::Op> clone() const final;

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const final;

  float beta() const { return _beta; }
  float threshold() const { return _threshold; }

private:
  float _beta;
  float _threshold;
};

class TorchSoftplusComputex : public popart::popx::EwuComputex {
public:
  TorchSoftplusComputex(float beta, float threshold)
      : _beta(beta), _threshold(threshold) {}

  void inplace(snap::program::Sequence &prog, snap::Graph &graph,
               const snap::Tensor &tensor, const poplar::DebugNameAndId &dnai,
               const std::string &prefix) const final;

  static std::unique_ptr<popart::popx::EwuComputex> get(float beta,
                                                        float threshold);

private:
  float _beta;
  float _threshold;
};

class TorchSoftplusOpx : public popart::popx::ElementWiseUnaryOutplaceOpx {
public:
  TorchSoftplusOpx(popart::Op *op, popart::popx::Devicex *devicex);
};

class TorchSoftplusInplaceOpx
    : public popart::popx::ElementWiseUnaryInplaceOpx {
public:
  TorchSoftplusInplaceOpx(popart::Op *op, popart::popx::Devicex *devicex);
};

class TorchSoftplusGradOpx : public popart::popx::PopOpx {
public:
  TorchSoftplusGradOpx(popart::Op *op, popart::popx::Devicex *devicex);
  void grow(snap::program::Sequence &prog) const final;

private:
  float _beta;
  float _threshold;
};

} // namespace poptorch_custom_ops

#endif
