// Copyright (c) 2021, Graphcore Ltd, All rights reserved.
#include <popart/builder.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>

#include "popart_compiler/CodeletsCompilation.hpp"
#include "popart_compiler/CompilerImpl.hpp"
#include "popart_compiler/CustomOps.hpp"

namespace {

struct BilinearParams {
  size_t input0;
  size_t input1;
  float lambda0;
  float lambda1;
};

float areaPixelComputeSourceIndex(float scale, size_t dst_index,
                                  bool align_corners, bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  }
  const float src_idx = static_cast<float>(scale * (dst_index + 0.5) - 0.5);
  // [Note] Follow Opencv resize logic:
  // We allow negative src_idx here and later will use
  //   dx = src_idx - floorf(src_idx)
  // to compute the "distance"(which affects weights).
  // For linear modes, weight distribution doesn't matter
  // for negative indices as they use 2 pixels to interpolate.
  // For example, [-1, 0], they both use pixel 0 value so it
  // doesn't affect if we bound the src_idx to 0 or not.
  // TODO(mihailp): Our current linear mode impls use unbound indices
  // where we should and then remove this cubic flag.
  // This matters in cubic mode, as we might need [-1, 0, 1, 2]
  // to interpolate and the weights can be affected.
  return (!cubic && src_idx < 0) ? 0.0f : src_idx;
}

BilinearParams computeSourceIndexAndLambda(const float scale,
                                           size_t output_index,
                                           size_t input_size,
                                           bool align_corners) {
  if (scale == 1.0) {
    // scale_factor = 1, simply copy
    return {output_index, output_index, 1.0, 0.0};
  }

  const float ratio = align_corners ? static_cast<float>(input_size - 1) /
                                          (scale * input_size - 1.0)
                                    : 1.0f / scale;

  const float real_input_index = areaPixelComputeSourceIndex(
      ratio, output_index, align_corners, /*cubic=*/false);
  const size_t index0 = static_cast<int64_t>(real_input_index);
  const size_t offset = (index0 < input_size - 1) ? 1 : 0;
  const float lambda1 = real_input_index - index0;

  return {index0, index0 + offset, 1.0f - lambda1, lambda1};
}

poplar::VertexRef connectVertex(
    poplar::Graph &graph, poplar::ComputeSet &cs, // NOLINT
    const std::string &vertexName,                // NOLINT
    const std::unordered_map<std::string, poplar::Tensor> &vars,
    const std::unordered_map<std::string, std::vector<poplar::Tensor>> &vectors,
    size_t tile) {
  poplar::VertexRef vtx = graph.addVertex(cs, vertexName);
  for (const auto &p : vars) {
    graph.connect(vtx[p.first], p.second);
  }
  for (const auto &p : vectors) {
    graph.connect(vtx[p.first], p.second);
  }
  graph.setPerfEstimate(vtx, 1);
  graph.setTileMapping(vtx, tile);
  return vtx;
}

using WeightKey = std::tuple<float, float, float, float>;
using WeightMap = std::map<WeightKey, poplar::Tensor>;

struct TileInputs {
  std::vector<poplar::Tensor> i00, i01, i10, i11, output;
  std::vector<std::vector<float>> weights;
  std::vector<poplar::Tensor> weights_t;
};

using MultipleTileMap = std::map<size_t, TileInputs>;

poplar::Tensor bilinearMap(poplar::Graph &graph,            // NOLINT
                           poplar::program::Sequence &prog, // NOLINT
                           const poplar::Tensor &input, float scale_factor,
                           const bool align_corners = false,
                           const poplar::DebugContext &dc = {}) {
  poputil::PoplibsOpDebugInfo const di(dc, DI_ARGS(input, scale_factor));

  const auto input_dims = input.shape();
  assert(input_dims.size() == 4); // NOLINT
  auto output_dims = input_dims;
  output_dims[2] =
      static_cast<std::size_t>(std::floor(output_dims[2] * scale_factor));
  output_dims[3] =
      static_cast<std::size_t>(std::floor(output_dims[3] * scale_factor));
  auto input_shuffled = input.dimShuffle({2, 3, 0, 1})
                            .reshape({input_dims[2], input_dims[3],
                                      input_dims[0] * input_dims[1]});
  std::vector<poplar::Tensor> i00s;

  std::vector<poplar::Tensor> i01s;

  std::vector<poplar::Tensor> i10s;

  std::vector<poplar::Tensor> i11s;
  std::vector<float> w00s;

  std::vector<float> w01s;

  std::vector<float> w10s;

  std::vector<float> w11s;
  for (size_t h = 0; h < output_dims[2]; ++h) {
    const BilinearParams params_h = computeSourceIndexAndLambda(
        scale_factor, h, input_dims[2], align_corners);
    for (size_t w = 0; w < output_dims[3]; ++w) {
      const BilinearParams params_w = computeSourceIndexAndLambda(
          scale_factor, w, input_dims[3], align_corners);
      w00s.push_back(params_h.lambda0 * params_w.lambda0);
      w01s.push_back(params_h.lambda0 * params_w.lambda1);
      w10s.push_back(params_h.lambda1 * params_w.lambda0);
      w11s.push_back(params_h.lambda1 * params_w.lambda1);
      i00s.push_back(input_shuffled[params_h.input0][params_w.input0]);
      i01s.push_back(input_shuffled[params_h.input0][params_w.input1]);
      i10s.push_back(input_shuffled[params_h.input1][params_w.input0]);
      i11s.push_back(input_shuffled[params_h.input1][params_w.input1]);
    }
  }
  poplar::Tensor const i00 = poplar::concat(i00s).reshape(
      {output_dims[2], output_dims[3], output_dims[0], output_dims[1]});
  poplar::Tensor const i01 = poplar::concat(i01s).reshape(
      {output_dims[2], output_dims[3], output_dims[0], output_dims[1]});
  poplar::Tensor const i10 = poplar::concat(i10s).reshape(
      {output_dims[2], output_dims[3], output_dims[0], output_dims[1]});
  poplar::Tensor const i11 = poplar::concat(i11s).reshape(
      {output_dims[2], output_dims[3], output_dims[0], output_dims[1]});
  const poplar::ArrayRef<float> w00_ref{w00s};

  const poplar::ArrayRef<float> w01_ref{w01s};

  const poplar::ArrayRef<float> w10_ref{w10s};

  const poplar::ArrayRef<float> w11_ref{w11s};
  poplar::Tensor const w00 = graph.addConstant(
      input.elementType(), {output_dims[2], output_dims[3], 1, 1}, w00_ref,
      {di, "w00"});
  poputil::mapTensorLinearly(graph, w00);
  poplar::Tensor const w01 = graph.addConstant(
      input.elementType(), {output_dims[2], output_dims[3], 1, 1}, w01_ref,
      {di, "w01"});
  poputil::mapTensorLinearly(graph, w01);
  poplar::Tensor const w10 = graph.addConstant(
      input.elementType(), {output_dims[2], output_dims[3], 1, 1}, w10_ref,
      {di, "w10"});
  poputil::mapTensorLinearly(graph, w10);
  poplar::Tensor const w11 = graph.addConstant(
      input.elementType(), {output_dims[2], output_dims[3], 1, 1}, w11_ref,
      {di, "w11"});
  poputil::mapTensorLinearly(graph, w11);
  poplar::Tensor const output = popops::map(
      graph,
      popops::expr::_1 * popops::expr::_2 +
          popops::expr::_3 * popops::expr::_4 +
          popops::expr::_5 * popops::expr::_6 +
          popops::expr::_7 * popops::expr::_8,
      {i00, w00, i01, w01, i10, w10, i11, w11}, prog, {di, "mapUpsampling"});
  return output.dimShuffle({2, 3, 0, 1});
}

using GradMultipleKey = std::pair<size_t, size_t>;
struct GradMultipleVal {
  float lambda0, lambda1;
  size_t h, w;
};
using GradMultipleMap = std::map<GradMultipleKey, std::vector<GradMultipleVal>>;

GradMultipleMap computeGradMap(size_t in_height, size_t in_width,
                               size_t out_height, size_t out_width,
                               float scale_factor, bool align_corners) {
  GradMultipleMap m;
  for (size_t h = 0; h < in_height; ++h) {
    const BilinearParams params_h =
        computeSourceIndexAndLambda(scale_factor, h, out_height, align_corners);
    for (size_t w = 0; w < in_width; ++w) {
      const BilinearParams params_w = computeSourceIndexAndLambda(
          scale_factor, w, out_width, align_corners);
      m[{params_h.input0, params_w.input0}].push_back(
          GradMultipleVal{params_h.lambda0, params_w.lambda0, h, w});
      m[{params_h.input0, params_w.input1}].push_back(
          GradMultipleVal{params_h.lambda0, params_w.lambda1, h, w});
      m[{params_h.input1, params_w.input0}].push_back(
          GradMultipleVal{params_h.lambda1, params_w.lambda0, h, w});
      m[{params_h.input1, params_w.input1}].push_back(
          GradMultipleVal{params_h.lambda1, params_w.lambda1, h, w});
    }
  }
  return m;
}

std::pair<std::vector<poplar::Tensor>, std::vector<float>>
computeInputsWeights(const std::vector<GradMultipleVal> &vals,
                     const poplar::Tensor &inputTensor) {
  std::vector<poplar::Tensor> inputs;
  std::vector<float> weights;
  size_t prev_w = -1;

  size_t prev_h = -1;
  for (const auto &v : vals) {
    const float weight = v.lambda0 * v.lambda1;
    if (weight > 0.0f) {
      if (v.h == prev_h && v.w == prev_w) {
        weights.back() += weight;
      } else {
        weights.push_back(weight);
        inputs.push_back(inputTensor[v.h][v.w]);
        prev_w = v.w;
        prev_h = v.h;
      }
    }
  }
  return std::make_pair(inputs, weights);
}

void splitIntervalMultiple(
    poplar::Graph &graph, poplar::ComputeSet &cs, // NOLINT
    size_t tile, const std::vector<poplar::Interval> &intervals,
    const poplar::Tensor &input, poplar::Tensor &output, // NOLINT
    const GradMultipleMap &m, const poplar::DebugContext &di) {
  const auto &full_interval = *intervals.begin();
  size_t start_block = full_interval.begin();
  const size_t block_size = output.shape()[2];
  while (start_block < full_interval.end()) {
    const size_t end_block =
        std::min(start_block + block_size - (start_block % block_size),
                 full_interval.end());
    std::vector<std::size_t> start_coords =
        poputil::unflattenIndex(output.shape(), start_block);
    std::vector<std::size_t> end_coords =
        poputil::unflattenIndex(output.shape(), end_block - 1);
    assert(start_coords[0] == end_coords[0]); // NOLINT
    assert(start_coords[1] == end_coords[1]); // NOLINT
    const auto iter = m.find({start_coords[0], start_coords[1]});
    assert(iter != m.end()); // NOLINT
    std::vector<poplar::Tensor> inputs;
    std::vector<float> weights;
    std::tie(inputs, weights) = computeInputsWeights(iter->second, input);
    poplar::Tensor weights_t = graph.addConstant(
        input.elementType(), {weights.size()}, poplar::ArrayRef<float>(weights),
        {di, "upsamplingGradWeights"});
    graph.setTileMapping(weights_t, tile);
    poplar::Tensor const full_input_t =
        poplar::concat(inputs).reshape({inputs.size(), block_size});
    poplar::Tensor const input_t = full_input_t.slice(
        {0, start_coords[2]}, {inputs.size(), end_coords[2] + 1});
    graph.setTileMapping(input_t, tile);
    poplar::Interval const interval{start_block, end_block};
    (void)connectVertex(
        graph, cs,
        poputil::templateVertex("BilinearGradVertex", input.elementType()),
        {{"out", output.flatten().slice(interval)},
         {"w", weights_t},
         {"input", input_t.transpose().flatten()}},
        {}, tile);

    start_block = end_block;
  }
}

void splitInterval(poplar::Graph &graph, poplar::ComputeSet &cs, // NOLINT
                   size_t tile, const std::vector<poplar::Interval> &intervals,
                   const poplar::Tensor &input,
                   poplar::Tensor &output, // NOLINT
                   const GradMultipleMap &m, const poplar::DebugContext &di) {
  const auto regions =
      poputil::splitRegionsBetweenWorkers(graph.getTarget(), intervals, 1);
  const size_t block_size = output.shape()[2];
  const auto &full_interval = *intervals.begin();
  std::vector<std::size_t> start_coords =
      poputil::unflattenIndex(output.shape(), full_interval.begin());
  std::vector<std::size_t> end_coords =
      poputil::unflattenIndex(output.shape(), full_interval.end() - 1);
  assert(start_coords[0] == end_coords[0]); // NOLINT
  assert(start_coords[1] == end_coords[1]); // NOLINT
  const auto iter = m.find({start_coords[0], start_coords[1]});
  assert(iter != m.end()); // NOLINT
  std::vector<poplar::Tensor> inputs;
  std::vector<float> weights;
  std::tie(inputs, weights) = computeInputsWeights(iter->second, input);
  poplar::Tensor weights_t = graph.addConstant(
      input.elementType(), {weights.size()}, poplar::ArrayRef<float>(weights),
      {di, "upsamplingGradWeights"});
  graph.setTileMapping(weights_t, tile);
  poplar::Tensor const full_input_t =
      poplar::concat(inputs).reshape({inputs.size(), block_size});
  for (const auto &r : regions) {
    assert(r.size() == 1); // NOLINT
    const auto &interval = *r.begin();
    start_coords = poputil::unflattenIndex(output.shape(), interval.begin());
    end_coords = poputil::unflattenIndex(output.shape(), interval.end() - 1);
    assert(start_coords[0] == end_coords[0]); // NOLINT
    assert(start_coords[1] == end_coords[1]); // NOLINT
    poplar::Tensor const input_t = full_input_t.slice(
        {0, start_coords[2]}, {inputs.size(), end_coords[2] + 1});
    graph.setTileMapping(input_t, tile);
    (void)connectVertex(
        graph, cs,
        poputil::templateVertex("BilinearGradVertex", input.elementType()),
        {{"out", output.flatten().slice(interval)},
         {"w", weights_t},
         {"input", input_t.transpose().flatten()}},
        {}, tile);
  }
}

void splitIntervalMultiplePixels(poplar::Graph &graph,   // NOLINT
                                 poplar::ComputeSet &cs, // NOLINT
                                 size_t tile,
                                 const std::vector<poplar::Interval> &intervals,
                                 const poplar::Tensor &input,
                                 poplar::Tensor &output, // NOLINT
                                 const GradMultipleMap &m,
                                 const poplar::DebugContext &di) {
  const size_t block_size = output.shape()[2];
  // each pixel is block_size in length
  const auto regions = poputil::splitRegionsBetweenWorkers(
      graph.getTarget(), intervals, block_size);
  for (const auto &r : regions) {
    assert(r.size() == 1); // NOLINT
    const auto &interval = *r.begin();
    assert((interval.size() % block_size) == 0); // NOLINT
    size_t start_block = interval.begin();
    std::vector<poplar::Tensor> full_inputs;
    std::vector<float> full_weights;
    std::vector<uint32_t> limits;
    while (start_block < interval.end()) {
      const size_t end_block = start_block + block_size;
      std::vector<std::size_t> start_coords =
          poputil::unflattenIndex(output.shape(), start_block);
      const std::vector<std::size_t> end_coords =
          poputil::unflattenIndex(output.shape(), end_block - 1);
      assert(start_coords[0] == end_coords[0]); // NOLINT
      assert(start_coords[1] == end_coords[1]); // NOLINT
      const auto iter = m.find({start_coords[0], start_coords[1]});
      assert(iter != m.end()); // NOLINT
      std::vector<poplar::Tensor> inputs;
      std::vector<float> weights;
      std::tie(inputs, weights) = computeInputsWeights(iter->second, input);
      limits.push_back(weights.size());
      std::copy(weights.begin(), weights.end(),
                std::back_inserter(full_weights));
      std::copy(inputs.begin(), inputs.end(), std::back_inserter(full_inputs));
      start_block = end_block;
    }
    poplar::Tensor weights_t = graph.addConstant(
        input.elementType(), {full_weights.size()},
        poplar::ArrayRef<float>(full_weights), {di, "upsamplingGradWeights"});
    graph.setTileMapping(weights_t, tile);
    poplar::Tensor limits_t = graph.addConstant(
        poplar::UNSIGNED_INT, {limits.size()},
        poplar::ArrayRef<unsigned int>(limits), {di, "upsamplingGradLimits"});
    graph.setTileMapping(limits_t, tile);

    poplar::Tensor const full_input_t =
        poplar::concat(full_inputs).reshape({full_inputs.size(), block_size});
    graph.setTileMapping(full_input_t, tile);
    assert(0 == (interval.size() % block_size)); // NOLINT
    (void)connectVertex(graph, cs,
                        poputil::templateVertex("BilinearGradMultipleVertex",
                                                input.elementType()),
                        {{"out", output.flatten().slice(interval)},
                         {"w", weights_t},
                         {"limits", limits_t},
                         {"input", full_input_t.transpose().flatten()}},
                        {}, tile);
  }
}

void processTile(poplar::Graph &graph, poplar::ComputeSet &cs, // NOLINT
                 size_t tile, const std::vector<poplar::Interval> &intervals,
                 const poplar::Tensor &input, poplar::Tensor &output, // NOLINT
                 const GradMultipleMap &m, const poplar::DebugContext &di) {
  assert(intervals.size() == 1); // NOLINT
  const poplar::Interval &interval = *intervals.begin();
  const size_t block_size = output.shape()[2];
  const size_t block_start = interval.begin() - (interval.begin() % block_size);
  const size_t aligned_size = interval.end() - block_start;
  const uint32_t nb_blocks = std::ceil(
      static_cast<float>(aligned_size / static_cast<float>(block_size)));
  if (nb_blocks == 1) {
    splitInterval(graph, cs, tile, intervals, input, output, m, di);
  } else {
    if (nb_blocks <= 6) {
      splitIntervalMultiple(graph, cs, tile, intervals, input, output, m, di);
    } else {
      splitIntervalMultiplePixels(graph, cs, tile, intervals, input, output, m,
                                  di);
    }
  }
}

using Mapping = std::vector<std::vector<poplar::Interval>>;

std::vector<Mapping> splitMapping(const Mapping &m, uint32_t partitions,
                                  uint32_t block_size) {
  if (partitions == 1) {
    return {m};
  }
  std::vector<Mapping> res(partitions);
  for (const auto &m_i : m) {
    const auto regions = poputil::splitRegions(m_i, block_size, partitions);
    for (size_t j = 0; j < regions.size(); ++j) {
      res[j].push_back(regions[j]);
    }
  }
  return res;
}

poplar::Tensor bilinearMapGrads(poplar::Graph &graph,            // NOLINT
                                poplar::program::Sequence &prog, // NOLINT
                                const poplar::Tensor &grad_output,
                                float scale_factor, bool align_corners,
                                uint32_t nb_splits = 0,
                                const poplar::DebugContext &dc = {}) {
  poputil::PoplibsOpDebugInfo const di(dc, DI_ARGS(grad_output, scale_factor));
  const auto grad_output_dims = grad_output.shape();
  assert(grad_output_dims.size() == 4); // NOLINT
  auto grad_input_dims = grad_output_dims;
  grad_input_dims[2] =
      static_cast<std::size_t>(std::floor(grad_output_dims[2] / scale_factor));
  grad_input_dims[3] =
      static_cast<std::size_t>(std::floor(grad_output_dims[3] / scale_factor));
  auto grad_input = graph.addVariable(
      grad_output.elementType(), grad_input_dims,
      {di, "gradientsInput_" + std::to_string(grad_input_dims[2])});
  auto grad_input_shuffled =
      grad_input.dimShuffle({2, 3, 0, 1})
          .reshape({grad_input_dims[2], grad_input_dims[3],
                    grad_input_dims[0] * grad_input_dims[1]});
  size_t grain_size = 1;
  const size_t nb_pixels = grad_input_dims[2] * grad_input_dims[3];
  const size_t num_tiles = graph.getTarget().getNumTiles();
  const size_t num_workers = graph.getTarget().getNumWorkerContexts();
  if (nb_pixels / num_tiles > num_workers) {
    grain_size = grad_output_dims[0] * grad_output_dims[1];
  }
  poputil::mapTensorLinearly(graph, grad_input_shuffled, 1, grain_size);
  auto grad_output_shuffled =
      grad_output.dimShuffle({2, 3, 0, 1})
          .reshape({grad_output_dims[2], grad_output_dims[3],
                    grad_output_dims[0] * grad_output_dims[1]});
  const GradMultipleMap m = computeGradMap(
      grad_output_dims[2], grad_output_dims[3], grad_input_dims[2],
      grad_input_dims[3], scale_factor, align_corners);
  const auto &full_mapping = graph.getTileMapping(grad_input_shuffled);
  if (nb_splits == 0) { // try to guess a good split
    nb_splits = 1;
    const uint32_t blocks_per_tile = std::ceil(static_cast<float>(nb_pixels) /
                                               static_cast<float>(num_tiles));
    if (blocks_per_tile > 6) {
      if (blocks_per_tile <= 12) {
        nb_splits = 2;
      } else {
        if (blocks_per_tile > 12) { // ?
          nb_splits = 3;
        }
      }
    }
  }
  const auto mappings = splitMapping(full_mapping, nb_splits,
                                     grad_output_dims[0] * grad_output_dims[1]);
  for (size_t split = 0; split < mappings.size(); ++split) {
    poplar::ComputeSet compute_set =
        graph.addComputeSet({di, "upsamplingGrad_" + std::to_string(split) +
                                     "_" + std::to_string(grad_input_dims[2])});
    const auto &mapping = mappings[split];
    for (size_t tile = 0; tile < mapping.size(); ++tile) {
      const auto &intervals = mapping[tile];
      if (!intervals.empty()) {
        processTile(graph, compute_set, tile, intervals, grad_output_shuffled,
                    grad_input_shuffled, m, di);
      }
    }
    prog.add(poplar::program::Execute(compute_set, di));
  }
  return grad_input;
}

// For training with a custom Op, four classes need to be implemented,
// one for each of:
// {forward, gradient} x {Op, Opx}.
//
// If only inference is required, then two classes need to be implemented:
// {forward} x {Op, Opx}.
//
// The Op is a poplar/hardware agnostic description of the computation.
// the Opx is the poplar implementation of the Op.
//
// We do training in this example, so the four classes implemented are:
//
class UpsampleOp;
class UpsampleGradOp;
class UpsampleOpx;
class UpsampleGradOpx;

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> makeUnique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

// The gradient Op
class UpsampleGradOp : public popart::Op {
public:
  explicit UpsampleGradOp(const UpsampleOp &fwdOp);

  std::unique_ptr<Op> clone() const final {
    return makeUnique<UpsampleGradOp>(*this);
  }

  // The output popart Tensor has the same inputInfo and numerical type
  // (i.e. the same TensorInfo) as the input Tensor. This function is
  // required for inputInfo/type inference
  //
  void setup() final {
    auto input_info = inInfo(0);
    assert(input_info.rank() == 4); // NOLINT
    auto batch_size = input_info.dim(0);
    auto channels = input_info.dim(1);
    auto height = input_info.dim(2);
    auto width = input_info.dim(3);
    const int64_t output_height =
        static_cast<int64_t>(std::floor(height / _scalingFactor));
    const int64_t output_width =
        static_cast<int64_t>(std::floor(width / _scalingFactor));

    outInfo(0).set(input_info.dataType(),
                   {batch_size, channels, output_height, output_width});
  }

  // function describing the inputs and output(s) of UpsampleGradOp
  // The Gradient Op which we are implementing (UpsampleGradOp) has 2 inputs.
  // The input at index 0 is:
  // the gradient of the 0'th output Tensor of the UpsampleOp.
  // The input at index 1 is :
  // the 0'th output Tensor of the UpsampleOp.
  // Supposing the UpsampleOp has input Tensor T0 and output Tensor T1,
  //
  //   input at index 0 (T0)
  //          |
  //        UpsampleOp
  //          |
  //   output at index 0 (T1)
  //
  // Then the picture described by the map below looks like,
  //
  //
  //    input at index 0 (gradient of T1)
  //         |   input at index 1 (T1)
  //         |     |
  //         |     |
  //        UpsampleGradOp
  //            |
  //            |
  //   output at index 0 (gradient of T0)
  //
  const std::vector<popart::GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<popart::GradInOutMapper> in_info = {
        {0, 0, popart::GradOpInType::GradOut},
        {1, 0, popart::GradOpInType::Out}};
    return in_info;
  }

  // The Grad Op only has one output, at index 0. The output at index 0
  // is the gradient of the input at index 0 of the UpsampleOp
  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> out_info = {{0, 0}};
    return out_info;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  float getScalingFactor() const { return _scalingFactor; }
  bool getAlignCorners() const { return _alignCorners; }
  // Implementation defined below
  void appendAttributes(popart::OpSerialiserBase &os) const override;

  // Implementation defined below
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

private:
  float _scalingFactor;
  bool _alignCorners;
};

// The forward Op
class UpsampleOp : public popart::Op {
public:
  UpsampleOp(const popart::OperatorIdentifier &_opid, float scalingFactor,
             bool alignCorners, const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), _scalingFactor{scalingFactor},
        _alignCorners(alignCorners) {}

  // same comment as for UpsampleGradOp, for running shape/type inference
  // "statically"
  void setup() override {
    auto input_info = inInfo(0);
    assert(input_info.rank() == 4); // NOLINT
    auto batch_size = input_info.dim(0);
    auto channels = input_info.dim(1);
    auto height = input_info.dim(2);
    auto width = input_info.dim(3);
    const int64_t output_height =
        static_cast<int64_t>(std::floor(height * _scalingFactor));
    const int64_t output_width =
        static_cast<int64_t>(std::floor(width * _scalingFactor));

    outInfo(0).set(input_info.dataType(),
                   {batch_size, channels, output_height, output_width});
  }

  std::unique_ptr<Op> clone() const final {
    return makeUnique<UpsampleOp>(*this);
  }

  // There is only one Gradient Op for UpsampleOp, a UpsampleGradOp
  // It is possible to have multiple Gradient Ops
  // (Conv has 2 in popart, one for weights and one for activations)
  //
  std::vector<std::unique_ptr<popart::Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> upops;        // NOLINT
    upops.emplace_back(new UpsampleGradOp(*this)); // NOLINT
    return upops;
  }
  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("scaling_factor", getScalingFactor());
    os.appendAttribute("align_corners", getAlignCorners());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("scaling_factor", getScalingFactor());
    os.appendAttribute("align_corners", getAlignCorners());
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  float getScalingFactor() const { return _scalingFactor; }
  bool getAlignCorners() const { return _alignCorners; }

private:
  float _scalingFactor;
  bool _alignCorners;
};

// describe the inputs and outputs that are supported by the operation
popart::OpDefinition::DataTypes t = {popart::DataType::FLOAT16,
                                     popart::DataType::FLOAT};

popart::OpDefinition upsample_op_def(
    {popart::OpDefinition::Inputs({{"input", t}}),
     popart::OpDefinition::Outputs({{"output", t}}),
     popart::OpDefinition::Attributes({{"scaling_factor", {"*"}},
                                       {"align_corners", {"*"}}})});

popart::OpCreator<UpsampleOp> upsample_op_creator(
    popart::OpDefinitions({{poptorch::poptorch_custom_ops::upsample_bilinear2d,
                            upsample_op_def}}),
    [](const popart::OpCreatorInfo &info) {
      // default scalingFactor is 2.0
      float const scaling_factor =
          info.attributes.getAttribute<popart::Attributes::Float>(
              "scaling_factor", 2.0f);
      int const align_corners =
          info.attributes.getAttribute<popart::Attributes::Int>("align_corners",
                                                                0);
      return std::make_unique<UpsampleOp>(info.opid, scaling_factor,
                                          align_corners, info.settings);
    },
    true);

// forward Opx (poplar implementation of the forward Op)
class UpsampleOpx : public popart::popx::Opx {
public:
  UpsampleOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    // not strictly necessary, we check that op is castable to a UpsampleOp *.
    verifyOp<UpsampleOp>(op,
                         poptorch::poptorch_custom_ops::upsample_bilinear2d);

    // Get around the ABI issues.
    auto managed_ptr = poptorch::popart_compiler::compileCustomCodeletIfNeeded(
        "UpsampleBilinear2dCodelets.inc.cpp", /*hw_only_codelet=*/false);
    const char *compiled_codelet_path =
        static_cast<const char *>(managed_ptr.get());
    graph().addCodelets(std::string(compiled_codelet_path));
  }

  void grow(poplar::program::Sequence &prog) const final {
    // Upsample the input. We create a poplar::Tensor of name outId(0)
    std::cerr << "Debug UpsampleOpx::grow\n";
    auto op = getOp<UpsampleOp>();
    const float scaling_factor = op.getScalingFactor();
    const bool align_corners = op.getAlignCorners();
    auto input = getInTensor(0);

    setOutTensor(
        0, bilinearMap(graph(), prog, input, scaling_factor, align_corners));
  }
};

// backward Opx (poplar implementation of the backward Op)
class UpsampleGradOpx : public popart::popx::Opx {
public:
  UpsampleGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<UpsampleGradOp>(
        op, poptorch::poptorch_custom_ops::upsample_bilinear2d_grad);
  }

  // Create the gradient poplar::Tensor, which is
  // 3 * input_to_upsample**2 * gradient_of_upsample_output
  void grow(poplar::program::Sequence &prog) const final {
    std::cerr << "Debug UpsampleGradOpx::grow\n";
    auto fwd_input = getInTensor(0);
    auto grad_out = getInTensor(1);

    auto op = getOp<UpsampleGradOp>();
    const float scaling_factor = op.getScalingFactor();
    const bool align_corners = op.getAlignCorners();
    setOutTensor(0, bilinearMapGrads(graph(), prog, grad_out, scaling_factor,
                                     align_corners));
  }
};

UpsampleGradOp::UpsampleGradOp(const UpsampleOp &fwdOp)
    : popart::Op(poptorch::poptorch_custom_ops::upsample_bilinear2d_grad,
                 fwdOp.settings),
      _scalingFactor{fwdOp.getScalingFactor()}, _alignCorners{
                                                    fwdOp.getAlignCorners()} {}

void UpsampleGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("scaling_factor", getScalingFactor());
  os.appendAttribute("align_corners", getAlignCorners());
}

void UpsampleGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("scaling_factor", getScalingFactor());
  os.appendAttribute("align_corners", getAlignCorners());
}

popart::popx::OpxCreator<UpsampleOpx>
    upsample_opx_creator(poptorch::poptorch_custom_ops::upsample_bilinear2d);
popart::popx::OpxCreator<UpsampleGradOpx> upsample_grad_opx_creator(
    poptorch::poptorch_custom_ops::upsample_bilinear2d_grad);

} // namespace
