// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <functional>

#include <ostream>
#include <popart/popx/opxmanager.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include "CustomOps.hpp"
#include "FastGatherLastDim.hpp"
#include "popart_compiler/CodeletsCompilation.hpp"
#include "popart_compiler/Utils.hpp"

namespace poptorch_custom_ops {

FastGatherLastDimOp::FastGatherLastDimOp(
    popart::OperatorIdentifier const &opid_,
    popart::Op::Settings const &settings_, std::string const &debug_str)
    : popart::Op(opid_, settings_) {
  this->_axis = -1;
  this->_debug_str = debug_str;
}

std::vector<std::unique_ptr<popart::Op>> FastGatherLastDimOp::getGradOps() {
  std::vector<std::unique_ptr<popart::Op>> upops;
  upops.emplace_back(std::make_unique<FastGatherLastDimGradOp>(*this));

  return upops;
}

std::unique_ptr<popart::Op> FastGatherLastDimOp::clone() const {
  return std::make_unique<FastGatherLastDimOp>(*this);
}

void FastGatherLastDimOp::setup() {
  if (poptorch::ipuModelEnvironmentVariableIsEnabled() ||
      poptorch::ipuSmallModelEnvironmentVariableIsEnabled()) {
    throw popart::error(
        "FastGatherLastDimOp requires hardware but IPU model is enabled");
  }

  popart::Shape data_shape = this->inInfo(0).shape();
  popart::Shape idx_shape = this->inInfo(1).shape();
  popart::Shape out_shape = data_shape;

  // idx rank and data rank should be the same
  if (data_shape.size() != idx_shape.size()) {
    throw popart::error(
        "FastGatherLastDimOp::setup(), "
        "Input and Index tensors do not have same rank in Op {}",
        this->getDebugStr());
  }

  // idx should have same dimensions as data except for last dim
  int data_rank = static_cast<int>(data_shape.size());
  for (unsigned i = 0; i < data_shape.size() - 1; i++) {
    if (idx_shape[i] != data_shape[i]) {
      throw popart::error("FastGatherLastDimOp::setup(), "
                          "Index tensor must have same dimensions as Input "
                          "except for last dim. Op {}",
                          this->getDebugStr());
    }
  }

  int axis = this->_axis;
  if (axis < 0) {
    axis = data_rank + axis;
  }
  for (unsigned i = 0; i < data_shape.size(); i++) {
    if (static_cast<unsigned>(axis) != i) {
      out_shape[i] = data_shape[i];
    }
  }

  out_shape[axis] = idx_shape[axis];
  this->_in_shape = data_shape;
  this->_out_shape = out_shape;
  this->outInfo(0) = {this->inInfo(0).dataType(), out_shape};
}

// register op
static popart::OpDefinition::DataTypes fast_gather_op_data_tensor_type = {
    popart::DataType::FLOAT16, popart::DataType::FLOAT};
static popart::OpDefinition::DataTypes fast_gather_op_idx_tensor_type = {
    popart::DataType::INT32, popart::DataType::INT16};

static popart::OpDefinition fast_gather_op_def(
    {popart::OpDefinition::Inputs({
         {"data", fast_gather_op_data_tensor_type},
         {"index", fast_gather_op_idx_tensor_type},
     }),
     popart::OpDefinition::Outputs({{"out", fast_gather_op_data_tensor_type}}),
     popart::OpDefinition::Attributes({})});

static popart::OpCreator<FastGatherLastDimOp> fast_gather_op_creator(
    popart::OpDefinitions({{poptorch_custom_ops::fast_gather_last_dim,
                            fast_gather_op_def}}),
    [](const popart::OpCreatorInfo &info) -> std::unique_ptr<popart::Op> {
      popart::OperatorIdentifier const &opid = info.opid;
      popart::Op::Settings const &settings = info.settings;
      popart::Attributes const &attr = info.attributes;
      std::string debug_str = attr.getAttribute<popart::Attributes::String>(
          "debug_str", "fast_gather_last_dim");
      return std::unique_ptr<popart::Op>(
          new FastGatherLastDimOp(opid, settings, debug_str));
    },
    true);

FastGatherLastDimOpx::FastGatherLastDimOpx(popart::Op *op,
                                           popart::popx::Devicex *devicex)
    : popart::popx::Opx(op, devicex) {
  verifyOp<FastGatherLastDimOp>(op, poptorch_custom_ops::fast_gather_last_dim);

  // Get around the ABI issues.
  auto managed_ptr = poptorch::compileCustomCodeletIfNeeded(
      "FastGatherLastDimFwdCodelets.inc.cpp", /*hw_only_codelet=*/true);
  const char *compiled_codelet_path =
      static_cast<const char *>(managed_ptr.get());
  graph().addCodelets(std::string(compiled_codelet_path));
}

void FastGatherLastDimOpx::grow(poplar::program::Sequence &prog) const {
  auto data_tensor = getInTensor(0);
  auto idx_tensor = getInTensor(1);

  FastGatherLastDimOp &fast_gather_last_dim_op = getOp<FastGatherLastDimOp>();
  popart::Shape fwd_op_out_shape = fast_gather_last_dim_op.getOutShape();

  std::vector<std::size_t> fwd_out_shape(fwd_op_out_shape.size());
  for (unsigned i = 0; i < fwd_op_out_shape.size(); i++) {
    fwd_out_shape[i] = fwd_op_out_shape[i];
  }

  poplar::Tensor out_tensor =
      addGraphProg(graph(), prog, data_tensor, idx_tensor, fwd_out_shape);

  setOutTensor(0, out_tensor);
}

poplar::Tensor FastGatherLastDimOpx::addGraphProg(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    poplar::Tensor const &data_tensor, poplar::Tensor const &idx_tensor,
    std::vector<std::size_t> const &fwd_out_shape) {

  poplar::Tensor output_tensor =
      graph.addVariable(data_tensor.elementType(), fwd_out_shape, "sel_out");
  auto target = graph.getTarget();
  unsigned num_tiles = target.getNumTiles();
  unsigned out_rank = idx_tensor.rank();

  std::size_t alloc_cnt = 1;
  std::size_t channel_cnt = 1;
  for (unsigned i = 0; i < out_rank; i++) {
    if (i < out_rank - 1) {
      alloc_cnt = alloc_cnt * fwd_out_shape[i];
    }
    if (i < out_rank - 2) {
      channel_cnt = channel_cnt * fwd_out_shape[i];
    }
  }
  auto in_shape = data_tensor.shape();
  auto out_shape = fwd_out_shape;

  poplar::ComputeSet gather_cs = graph.addComputeSet("FastGatherCS");
  std::vector<unsigned> tile_start(num_tiles, 0);
  std::vector<unsigned> tile_count(num_tiles, 0);

  poplar::Tensor data_tensor_clone = graph.clone(data_tensor);
  poplar::Tensor data_tensor_reshape =
      data_tensor_clone.reshape({alloc_cnt, in_shape[out_rank - 1]});

  poplar::Tensor idx_tensor_clone = graph.clone(idx_tensor);
  poplar::Tensor idx_tensor_reshape =
      idx_tensor_clone.reshape({alloc_cnt, out_shape[out_rank - 1]});
  poplar::Tensor result_tensor_reshape =
      output_tensor.reshape({alloc_cnt, out_shape[out_rank - 1]});

  std::size_t tile_idx_last = 1;
  for (std::size_t i = 0; i < alloc_cnt; ++i) {
    std::size_t idx = (i * num_tiles) / alloc_cnt;
    graph.setTileMapping(data_tensor_reshape[i], idx);
    graph.setTileMapping(idx_tensor_reshape[i], idx);
    graph.setTileMapping(result_tensor_reshape[i], idx);
    if (tile_idx_last != idx) {
      tile_start[idx] = i;
    }
    tile_count[idx] += 1;
    tile_idx_last = idx;
  }
  prog.add(poplar::program::Copy(data_tensor, data_tensor_clone));
  prog.add(poplar::program::Copy(idx_tensor, idx_tensor_clone));

  for (unsigned i = 0; i < num_tiles; ++i) {
    if (0 == tile_count[i]) {
      continue;
    }

    poplar::VertexRef gather_vertex = graph.addVertex(
        gather_cs,
        poputil::templateVertex("FastGatherVertex", data_tensor.elementType(),
                                idx_tensor.elementType()),
        {{"data_", data_tensor_reshape.slice(tile_start[i],
                                             tile_start[i] + tile_count[i])},
         {"idx_", idx_tensor_reshape.slice(tile_start[i],
                                           tile_start[i] + tile_count[i])},
         {"result_", result_tensor_reshape.slice(
                         tile_start[i], tile_start[i] + tile_count[i])}});
    graph.setTileMapping(gather_vertex, i);
    graph.setInitialValue(gather_vertex["dst_shape_"], out_shape);
  }

  prog.add(poplar::program::Execute(gather_cs));

  return output_tensor;
}

FastGatherLastDimGradOp::FastGatherLastDimGradOp(
    const FastGatherLastDimOp &fwdOp)
    : popart::Op(poptorch_custom_ops::fast_gather_last_dim_grad,
                 fwdOp.getSettings()) {
  this->_axis = -1;
  this->_fwd_in_shape = fwdOp.getInShape();
  this->_debug_str = fwdOp.getDebugStr();
}

std::unique_ptr<popart::Op> FastGatherLastDimGradOp::clone() const {
  return std::make_unique<FastGatherLastDimGradOp>(*this);
}

FastGatherLastDimGradOpx::FastGatherLastDimGradOpx(
    popart::Op *op, popart::popx::Devicex *devicex)
    : popart::popx::Opx(op, devicex) {
  verifyOp<FastGatherLastDimGradOp>(
      op, poptorch_custom_ops::fast_gather_last_dim_grad);

  // Get around the ABI issues.
  auto managed_ptr = poptorch::compileCustomCodeletIfNeeded(
      "FastGatherLastDimBwdCodelets.inc.cpp", /*hw_only_codelet=*/true);
  const char *compiled_codelet_path =
      static_cast<const char *>(managed_ptr.get());
  graph().addCodelets(std::string(compiled_codelet_path));
}

void FastGatherLastDimGradOpx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor grad_output_tensor = getInTensor(0);
  poplar::Tensor idx_tensor = getInTensor(1);

  FastGatherLastDimGradOp &grad_op = getOp<FastGatherLastDimGradOp>();
  popart::Shape fwd_in_shape = grad_op.getFwdInShape();
  std::vector<std::size_t> fwd_in_shape_2(fwd_in_shape.size());
  for (unsigned i = 0; i < fwd_in_shape.size(); i++) {
    fwd_in_shape_2[i] = static_cast<std::size_t>(fwd_in_shape[i]);
  }

  auto zero = getScalarVariable(grad_output_tensor.elementType(), "zero");
  graph().setInitialValue(zero, 0);
  auto output = zero;
  for (unsigned i = 0; i < fwd_in_shape.size(); ++i) {
    output = output.expand({0});
  }
  for (unsigned i = 0; i < fwd_in_shape.size(); ++i) {
    output = output.broadcast(static_cast<unsigned>(fwd_in_shape[i]), i);
  }

  auto out_tensor = cloneNcopy(prog, output);

  poplar::Tensor grad_input_tensor =
      addGraphProg(graph(), prog, grad_output_tensor, out_tensor,
                   fwd_in_shape_2, idx_tensor);

  setOutTensor(0, out_tensor);
}

poplar::Tensor FastGatherLastDimGradOpx::addGraphProg(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    poplar::Tensor const &grad_output_tensor, poplar::Tensor &grad_input_tensor,
    std::vector<std::size_t> const &fwd_in_shape,
    poplar::Tensor const &idx_tensor) {

  auto target = graph.getTarget();
  unsigned num_tiles = target.getNumTiles();
  unsigned grad_output_rank = grad_output_tensor.rank();

  std::size_t alloc_cnt = 1;
  std::size_t channel_cnt = 1;
  for (unsigned i = 0; i < grad_output_rank; i++) {
    if (i < grad_output_rank - 1) {
      alloc_cnt = alloc_cnt * grad_output_tensor.dim(i);
    }
    if (i < grad_output_rank - 2) {
      channel_cnt = channel_cnt * grad_output_tensor.dim(i);
    }
  }
  auto grad_output_shape = grad_output_tensor.shape();
  auto grad_input_shape = fwd_in_shape;

  poplar::ComputeSet gather_grad_cs = graph.addComputeSet("FastGatherGradCS");
  std::vector<unsigned> tile_start(num_tiles, 0);
  std::vector<unsigned> tile_count(num_tiles, 0);

  poplar::Tensor grad_output_tensor_clone = graph.clone(grad_output_tensor);

  poplar::Tensor grad_output_tensor_reshape = grad_output_tensor_clone.reshape(
      {alloc_cnt, grad_output_shape[grad_output_rank - 1]});

  poplar::Tensor idx_tensor_clone = graph.clone(idx_tensor);
  poplar::Tensor idx_tensor_reshape = idx_tensor_clone.reshape(
      {alloc_cnt, grad_output_shape[grad_output_rank - 1]});

  poplar::Tensor grad_input_tensor_reshape = grad_input_tensor.reshape(
      {alloc_cnt, grad_input_shape[grad_output_rank - 1]});

  std::size_t tile_idx_last = 1;
  for (std::size_t i = 0; i < alloc_cnt; ++i) {
    std::size_t idx = (i * num_tiles) / alloc_cnt;
    graph.setTileMapping(grad_output_tensor_reshape[i], idx);
    graph.setTileMapping(idx_tensor_reshape[i], idx);
    graph.setTileMapping(grad_input_tensor_reshape[i], idx);
    if (tile_idx_last != idx) {
      tile_start[idx] = i;
    }
    tile_count[idx] += 1;
    tile_idx_last = idx;
  }
  prog.add(poplar::program::Copy(idx_tensor, idx_tensor_clone));
  prog.add(poplar::program::Copy(grad_output_tensor, grad_output_tensor_clone));

  for (unsigned i = 0; i < num_tiles; ++i) {
    if (0 == tile_count[i]) {
      continue;
    }

    poplar::VertexRef gather_vertex = graph.addVertex(
        gather_grad_cs,
        poputil::templateVertex("FastGatherGradVertex",
                                grad_output_tensor.elementType(),
                                idx_tensor.elementType()),
        {{"grad_out_", grad_output_tensor_reshape.slice(
                           tile_start[i], tile_start[i] + tile_count[i])},
         {"idx_", idx_tensor_reshape.slice(tile_start[i],
                                           tile_start[i] + tile_count[i])},
         {"grad_in_", grad_input_tensor_reshape.slice(
                          tile_start[i], tile_start[i] + tile_count[i])}});
    graph.setTileMapping(gather_vertex, i);
    graph.setInitialValue(gather_vertex["grad_out_shape_"], grad_output_shape);
    graph.setInitialValue(gather_vertex["grad_in_shape_"], grad_input_shape);
  }

  prog.add(poplar::program::Execute(gather_grad_cs));

  return grad_input_tensor;
}

namespace {
popart::popx::OpxCreator<FastGatherLastDimOpx>
    fast_gather_last_dim_opx(poptorch_custom_ops::fast_gather_last_dim);
popart::popx::OpxCreator<FastGatherLastDimGradOpx>
    fast_gather_last_dim_grad_opx(
        poptorch_custom_ops::fast_gather_last_dim_grad);
} // namespace

} // namespace poptorch_custom_ops
