// Copyright (c) 2021, Graphcore Ltd, All rights reserved.

// TODO(T70346): snap:: API is deprecated
#pragma GCC diagnostic warning "-Wdeprecated-declarations"

#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "popart_compiler/CustomOps.hpp"
#include <popart/op.hpp>
#include <popart/op/gather.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/vendored/optional.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/Zero.hpp>

namespace poptorch {
namespace poptorch_custom_ops {

class EmbeddingGradOp;

// EmbeddingOp needs to be convertible to popart::GatherOp so that the tied
// gather pattern can match this implementation.
class EmbeddingOp : public popart::GatherOp {
public:
  EmbeddingOp(const popart::OperatorIdentifier &_opid,
              const nonstd::optional<int64_t> &padding_idx,
              const nonstd::optional<float> &available_memory_proportion_,
              const popart::Op::Settings &settings_)
      : popart::GatherOp(_opid, /*axis=*/0, settings_,
                         available_memory_proportion_),
        _padding_idx(padding_idx) {}

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<EmbeddingOp>(*this);
  }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() final {
    std::vector<std::unique_ptr<popart::Op>> result;
    result.emplace_back(std::make_unique<EmbeddingGradOp>(*this));
    return result;
  }

  static popart::InIndex weightInIndex() { return 0; }
  static popart::InIndex indicesInIndex() { return 1; }
  static popart::OutIndex outIndex() { return 0; }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const final {
    popart::GatherOp::appendOutlineAttributes(os);
    os.appendAttribute("padding_idx", paddingIndex());
  }

  nonstd::optional<int64_t> paddingIndex() const { return _padding_idx; }

private:
  nonstd::optional<int64_t> _padding_idx;
};

class EmbeddingGradOp : public popart::Op {
public:
  explicit EmbeddingGradOp(const EmbeddingOp &fwd_op)
      : popart::Op(poptorch_custom_ops::embedding_grad, fwd_op.getSettings()),
        _padding_idx(fwd_op.paddingIndex()),
        _available_memory_proportion(fwd_op.getAvailableMemoryProportion()),
        _wieght_info(fwd_op.inInfo(EmbeddingOp::weightInIndex())) {}

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<EmbeddingGradOp>(*this);
  }

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const final {
    static const std::vector<popart::GradInOutMapper> info = {
        {gradInIndex(), EmbeddingOp::outIndex(), popart::GradOpInType::GradOut},
        {indicesInIndex(), EmbeddingOp::indicesInIndex(),
         popart::GradOpInType::In}};

    return info;
  }

  const std::map<int, int> &gradOutToNonGradIn() const final {
    static const std::map<int, int> out = {
        {gradOutIndex(), EmbeddingOp::weightInIndex()}};

    return out;
  }

  void setup() final { outInfo(gradOutIndex()) = _wieght_info; }

  static popart::InIndex gradInIndex() { return 0; }
  static popart::InIndex indicesInIndex() { return 1; }
  static popart::OutIndex gradOutIndex() { return 0; }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const final {
    popart::Op::appendOutlineAttributes(os);
    os.appendAttribute("padding_idx", paddingIndex());
    os.appendAttribute(popart::sAvailMemAttribute, availableMemoryProportion());
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  nonstd::optional<int64_t> paddingIndex() const { return _padding_idx; }

  nonstd::optional<float> availableMemoryProportion() const {
    return _available_memory_proportion;
  }

private:
  nonstd::optional<int64_t> _padding_idx;
  nonstd::optional<float> _available_memory_proportion;
  popart::TensorInfo _wieght_info;
};

namespace {
popart::OpDefinition::DataTypes weight_dtypes = {
    popart::DataType::UINT8,   popart::DataType::UINT16,
    popart::DataType::UINT32,  popart::DataType::UINT64,
    popart::DataType::INT8,    popart::DataType::INT16,
    popart::DataType::INT32,   popart::DataType::INT64,
    popart::DataType::FLOAT16, popart::DataType::FLOAT};

popart::OpDefinition::DataTypes indices_dtypes = {
    popart::DataType::UINT8,  popart::DataType::UINT16,
    popart::DataType::UINT32, popart::DataType::UINT64,
    popart::DataType::INT8,   popart::DataType::INT16,
    popart::DataType::INT32,  popart::DataType::INT64};

popart::OpDefinition
    embedding_def({popart::OpDefinition::Inputs({{"weight", weight_dtypes},
                                                 {"indices", indices_dtypes}}),
                   popart::OpDefinition::Outputs({{"output", weight_dtypes}}),
                   popart::OpDefinition::Attributes({
                       {"padding_idx", {"*"}},
                   })});

popart::OpCreator<EmbeddingOp> embedding_creator(
    popart::OpDefinitions({{poptorch_custom_ops::embedding, embedding_def}}),
    [](const popart::OpCreatorInfo &info) {
      nonstd::optional<int64_t> padding_idx;

      if (info.attributes.hasAttribute("padding_idx")) {
        padding_idx = info.attributes.getAttribute<popart::Attributes::Int>(
            "padding_idx");
      }

      nonstd::optional<float> available_memory_proportion;

      if (info.attributes.hasAttribute(popart::sAvailMemAttribute)) {
        available_memory_proportion =
            info.attributes.getAttribute<popart::Attributes::Float>(
                popart::sAvailMemAttribute);
      }

      return std::unique_ptr<popart::Op>(new EmbeddingOp(
          info.opid, padding_idx, available_memory_proportion, info.settings));
    },
    true);

} // namespace

class EmbeddingOpx : public popart::popx::PopOpx {
public:
  EmbeddingOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::PopOpx(op, devicex) {
    verifyOp<EmbeddingOp>(op, {poptorch_custom_ops::embedding});

    // We always want the EmbeddingOpx to layout its inputs
    inputCreatorPriority = std::numeric_limits<double>::max();

    auto options = popart::popx::createSlicePlanOptions(
        popart::popx::SlicePlanUsedFor::Slice,
        getOp<EmbeddingOp>().getAvailableMemoryProportion());

    _plan = popart::popx::createSlicePlan(
        graph(), inInfo(EmbeddingOp::weightInIndex()),
        inInfo(EmbeddingOp::indicesInIndex()), options, /*axis=*/0);
  }

  void grow(snap::program::Sequence &prog) const final {
    auto weight = getInTensor(EmbeddingOp::weightInIndex());
    auto indices = getInTensor(EmbeddingOp::indicesInIndex());

    // Assume non-negative indices.
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
    indices = indices.flatten();
    indices = indices.expand({1});

    auto result = popops::multiSlice(
        graph().getPoplarGraph(), weight.getPoplarTensor(),
        indices.getPoplarTensor(), {0}, {1}, prog.getPoplarSequence(), _plan,
        poplar::OptionFlags());

    result = result.reshape(outInfo(EmbeddingOp::outIndex()).shape_szt());
    setOutTensor(EmbeddingOp::outIndex(), snap::Tensor{result, graph()});
  }

  snap::Tensor
  createInputTensor(popart::InIndex index,
                    const poplar::DebugNameAndId &dnai) const final {
    if (index != EmbeddingOp::weightInIndex() &&
        index != EmbeddingOp::indicesInIndex()) {
      throw popart::error(
          "EmbeddingOpx::createInputTensor : Invalid index = {}", index);
    }

    if (index == EmbeddingOp::weightInIndex()) {
      const auto &weight_info = inInfo(index);
      auto weight = popops::createSliceableTensor(
          graph().getPoplarGraph(), popart::popx::popType(weight_info),
          weight_info.shape_szt(), {0}, {1}, _plan, poplar::OptionFlags(),
          dnai);

      return snap::Tensor{weight, graph()};
    }

    const auto &indices_info = inInfo(index);
    auto num_lookups = static_cast<std::size_t>(indices_info.nelms());
    auto indices =
        popops::createIndicesTensor(graph().getPoplarGraph(), {0}, num_lookups,
                                    _plan, poplar::OptionFlags(), dnai);

    indices = indices.reinterpret(popart::popx::popType(indices_info));
    indices = indices.reshape(indices_info.shape_szt());
    return snap::Tensor{indices, graph()};
  }

  popart::popx::InputCreatorType
  getInputCreatorType(popart::InIndex index) const final {
    if (index == EmbeddingOp::weightInIndex() ||
        index == EmbeddingOp::indicesInIndex()) {
      return popart::popx::InputCreatorType::CanCreate;
    }

    return PopOpx::getInputCreatorType(index);
  }

  std::set<popart::TensorId>
  mustExistBeforeCreate(popart::InIndex index) const final {
    (void)index; // unused
    return {};
  }

private:
  popops::SlicePlan _plan;
};

class EmbeddingGradOpx : public popart::popx::PopOpx {
public:
  EmbeddingGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::PopOpx(op, devicex) {
    verifyOp<EmbeddingGradOp>(op, {poptorch_custom_ops::embedding_grad});

    // We always want the EmbeddingGradOpx to layout its inputs
    inputCreatorPriority = std::numeric_limits<double>::max();

    auto grad_op = getOp<EmbeddingGradOp>();
    _padding_idx = grad_op.paddingIndex();

    auto options = popart::popx::createSlicePlanOptions(
        popart::popx::SlicePlanUsedFor::UpdateAdd,
        grad_op.availableMemoryProportion());

    _plan = popart::popx::createSlicePlan(
        graph(), outInfo(EmbeddingGradOp::gradOutIndex()),
        inInfo(EmbeddingGradOp::indicesInIndex()), options, /*axis=*/0);
  }

  void grow(snap::program::Sequence &prog) const final {
    auto grad_in = getInTensor(EmbeddingGradOp::gradInIndex());
    auto indices = getInTensor(EmbeddingGradOp::indicesInIndex());
    auto output_shape = outInfo(EmbeddingGradOp::gradOutIndex()).shape_szt();

    auto out = popops::createSliceableTensor(
        graph().getPoplarGraph(), grad_in.elementType(), output_shape, {0}, {1},
        _plan, poplar::OptionFlags(), debugContext("embedding_grad_out"));

    popops::zero(graph().getPoplarGraph(), out, prog.getPoplarSequence(),
                 debugContext("zero"));

    auto scale = graph().getPoplarGraph().addConstant(
        grad_in.elementType(), {}, 1.0f, debugContext("const_1"));
    graph().getPoplarGraph().setTileMapping(scale, 0);

    auto inputs = popart::popx::GatherGradOpx::handleNDMultiUpdate(
        out, grad_in.getPoplarTensor(), indices.getPoplarTensor(), 0);
    auto &target_nd = std::get<0>(inputs);
    auto &update_nd = std::get<1>(inputs);
    auto &indices_nd = std::get<2>(inputs);

    popops::multiUpdateAdd(
        graph().getPoplarGraph(), target_nd, update_nd, indices_nd, scale, {0},
        {1}, prog.getPoplarSequence(), _plan, poplar::OptionFlags(),
        debugContext("embedding_grad"));

    if (_padding_idx) {
      auto start = static_cast<std::size_t>(*_padding_idx);
      auto padding = out.slice(start, start + 1, 0);
      popops::zero(graph().getPoplarGraph(), padding, prog.getPoplarSequence(),
                   debugContext("zero_padding_idx"));
    }

    setOutTensor(EmbeddingGradOp::gradOutIndex(), snap::Tensor{out, graph()});
  }

  snap::Tensor
  createInputTensor(popart::InIndex index,
                    const poplar::DebugNameAndId &dnai) const final {
    if (index != EmbeddingGradOp::gradInIndex() &&
        index != EmbeddingGradOp::indicesInIndex()) {
      throw popart::error(
          "EmbeddingGradOpx::createInputTensor : Invalid index = {}", index);
    }

    if (index == EmbeddingGradOp::gradInIndex()) {
      const auto &grad_info = inInfo(index);
      auto weight = popops::createSliceableTensor(
          graph().getPoplarGraph(), popart::popx::popType(grad_info),
          grad_info.shape_szt(), {0}, {1}, _plan, poplar::OptionFlags(), dnai);

      return snap::Tensor{weight, graph()};
    }

    const auto &indices_info = inInfo(index);
    auto num_lookups = static_cast<std::size_t>(indices_info.nelms());
    auto indices =
        popops::createIndicesTensor(graph().getPoplarGraph(), {0}, num_lookups,
                                    _plan, poplar::OptionFlags(), dnai);

    indices = indices.reinterpret(popart::popx::popType(indices_info));
    indices = indices.reshape(indices_info.shape_szt());
    return snap::Tensor{indices, graph()};
  }

  popart::popx::InputCreatorType
  getInputCreatorType(popart::InIndex index) const final {
    if (index == EmbeddingGradOp::gradInIndex() ||
        index == EmbeddingGradOp::indicesInIndex()) {
      return popart::popx::InputCreatorType::CanCreate;
    }

    return PopOpx::getInputCreatorType(index);
  }

  std::set<popart::TensorId>
  mustExistBeforeCreate(popart::InIndex index) const final {
    (void)index; // unused
    return {};
  }

private:
  nonstd::optional<int64_t> _padding_idx;
  popops::SlicePlan _plan;
};

namespace {
popart::popx::OpxCreator<EmbeddingOpx>
    embedding_opx(poptorch_custom_ops::embedding);
popart::popx::OpxCreator<EmbeddingGradOpx>
    embedding_grad_opx(poptorch_custom_ops::embedding_grad);
} // namespace

} // namespace poptorch_custom_ops
} // namespace poptorch
