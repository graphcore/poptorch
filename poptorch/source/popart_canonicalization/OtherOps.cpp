// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <c10/core/ScalarType.h>

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "EinsumOp.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "ScatterReduction.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include <ATen/ATen.h>

namespace poptorch {
namespace {
torch::jit::Node *einsumHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::einsum(string equation, Tensor[] tensors) -> Tensor

  // Einstein summation convention equation
  const std::string eq = constantToString(node->input(0)->node());
  // List of inputs to perform the operation on
  const std::vector<torch::jit::Value *> tensors =
      handleTensorList(node->input(1)->node());

  const std::vector<std::int64_t> output_shape =
      shapeFromTensor(node->output());
  EinsumOp einsum(eq, tensors);
  return einsum.create(graph, output_shape);
}

torch::jit::Node *meshgridHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::meshgrid(Tensor[] tensors) -> Tensor[]

  const std::vector<torch::jit::Value *> tensors =
      handleTensorList(node->input(0)->node());

  std::vector<std::int64_t> expand_shape;
  expand_shape.reserve(tensors.size());
  for (torch::jit::Value *tensor : tensors) {
    // Each tensor is 1D so the shape is just the first dim
    expand_shape.push_back(shapeFromTensor(tensor)[0]);
  }

  std::vector<torch::jit::Value *> grids;
  for (std::size_t i = 0; i < tensors.size(); i++) {
    std::vector<std::int64_t> shape(tensors.size(), 1);
    shape[i] = -1;
    // Reshape 1D tensor to rank N, N = number of tensors, such that
    // all but the ith dimension is a singleton
    torch::jit::Node *reshaped = createReshape(graph, tensors[i], shape);
    // Expand over the dimensions of all other tensors
    torch::jit::Node *expanded =
        createExpand(graph, {reshaped->output(),
                             intVectorToIrConstant(graph, expand_shape)});
    grids.push_back(expanded->output());
  }

  return createAndInsertNode(graph, at::prim::ListConstruct, grids);
}

torch::jit::Node *cartesianProdHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  // aten::cartesian_prod(Tensor[] tensors) -> Tensor

  const std::vector<torch::jit::Value *> tensors =
      handleTensorList(node->input(0)->node());

  if (tensors.size() == 1) {
    return tensors[0]->node();
  }

  auto meshgrid_handler = getHandler(c10::aten::meshgrid);
  auto stack_handler = getHandler(c10::aten::stack);

  torch::jit::Node *grids =
      createHandlerOperation(graph, meshgridHandler, {node->input(0)});

  std::vector<torch::jit::Value *> grids_vector = handleTensorList(grids);

  for (torch::jit::Value *&grid : grids_vector) {
    // Flatten into 1 x N
    torch::jit::Node *flatten = createFlatten(graph, {grid}, 0);
    // Squeeze the first dimension
    flatten = createSqueeze(graph, {flatten->output()}, {0});
    grid = flatten->output();
  }

  torch::jit::Node *grid_list =
      createAndInsertNode(graph, at::prim::ListConstruct, grids_vector);

  // Stack 1D tensors along dimension 1
  return createHandlerOperation(
      graph, stack_handler, {grid_list->output(), wrapInConstant1D(graph, 1)});
}

torch::jit::Node *tensordotHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::tensordot(Tensor self, Tensor other, int[] dims_self,
  //                 int[] dims_other) -> Tensor

  torch::jit::Value *x1 = node->input(0);
  torch::jit::Value *x2 = node->input(1);
  std::vector<std::int64_t> rdims_x1 =
      constantToLongVec(node->input(2)->node());
  const std::vector<std::int64_t> rdims_x2 =
      constantToLongVec(node->input(3)->node());

  // rdims_prod (default = 1 with no reduction)
  std::int64_t rdims_prod = 1;

  const std::vector<std::int64_t> shape_x1 = shapeFromTensor(x1);
  const std::vector<std::int64_t> shape_x2 = shapeFromTensor(x2);

  // Original permutation
  std::vector<std::int64_t> p1 = shape_x1;
  std::iota(p1.begin(), p1.end(), 0);
  std::vector<std::int64_t> p2 = shape_x2;
  std::iota(p2.begin(), p2.end(), 0);

  const std::size_t n_dims_x1 = p1.size();
  const std::size_t n_dims_x2 = p2.size();
  const std::size_t n_rdims = rdims_x1.size();

  // Negative (relative) indexing -> absolute indexing
  for (std::int64_t &rdim : rdims_x1) {
    if (rdim < 0) {
      rdim += n_dims_x1;
    }
  }

  std::vector<bool> rdims_x1_bs(n_dims_x1);
  std::vector<bool> rdims_x2_bs(n_dims_x2);
  for (std::size_t i = 0; i < n_rdims; i++) {
    rdims_x1_bs[rdims_x1[i]] = true;
    rdims_x2_bs[rdims_x2[i]] = true;
    // prod(rdims_x1) == prod(rdims_x2) so just use x1
    rdims_prod *= shape_x1[rdims_x1[i]];
  }

  // Permutes x according to existing permutation vector p and bitset bs. If
  // should_partition_front == true, elements of p are moved to the front
  // if the corresponding bool in bs == true. Otherwise, they are moved to
  // the back. The relative order of other elements must not change.
  const auto fn_partition_permute = [&](torch::jit::Value *x, auto &p,
                                        const auto &bs,
                                        bool should_partition_front) {
    std::stable_partition(p.begin(), p.end(), [&](std::int64_t n) {
      return bs[n] == should_partition_front;
    });
    return createTranspose(graph, {x}, p);
  };

  // Permute x1 so that rdims_x1 are the last dims
  torch::jit::Node *p_x1 = fn_partition_permute(x1, p1, rdims_x1_bs, false);

  // Reshape to (-1, rdims_prod(rdims))
  torch::jit::Node *p_x1_mat =
      createReshape(graph, p_x1->output(), {-1, rdims_prod});

  // Permute x2 so that rdims_x2 are the first dims
  torch::jit::Node *p_x2 = fn_partition_permute(x2, p2, rdims_x2_bs, true);

  // Reshape to (rdims_prod(rdims), -1)
  torch::jit::Node *p_x2_mat =
      createReshape(graph, p_x2->output(), {rdims_prod, -1});

  // Matmul -> (unreduced_x1, unreduced_x2)
  torch::jit::Node *mm =
      createMatmul(graph, {p_x1_mat->output(), p_x2_mat->output()});

  std::vector<std::int64_t> new_shape;
  new_shape.reserve(n_dims_x1 + n_dims_x2);
  for (std::size_t i = 0; i < n_dims_x1; i++) {
    if (!rdims_x1_bs[i]) {
      new_shape.push_back(shape_x1[i]);
    }
  }
  for (std::size_t i = 0; i < n_dims_x2; i++) {
    if (!rdims_x2_bs[i]) {
      new_shape.push_back(shape_x2[i]);
    }
  }

  // Restore flattened dims
  return createReshape(graph, mm->output(), new_shape);
}

bool isIndexBroadcastEnabled(torch::jit::Node *node) {
  static const auto bcast_attr = c10::Symbol::attr("enable_index_broadcast");

  return node->hasAttribute(bcast_attr) ? static_cast<bool>(node->i(bcast_attr))
                                        : false;
}

torch::jit::Node *scatterAddHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {

  static constexpr std::int32_t sum_reduce =
      static_cast<std::int32_t>(ScatterReduction::Sum);

  auto *output = node->input(0);
  auto *index = node->input(2);
  auto *src = node->input(3);
  const auto src_type = src->type()->expect<c10::TensorType>();
  const auto axis = handleDimensionParam(node->input(1), src_type);
  const auto shape = shapeFromTensor(node->output());
  const auto axissize = shape.at(axis);
  const auto enable_index_broadcast = isIndexBroadcastEnabled(node);

  if (isTensorConstant(output->node())) {
    // output may have been generated by calling zeros(...) and at this point
    // in the canonicalization the node is represented as a tensor constant.
    auto out_tensor = getNodeTensorAttrValue(output->node());
    const auto scalar_zero = at::zeros(1, out_tensor.dtype());
    const bool all_zeros = at::all(out_tensor.eq(scalar_zero)).item().toBool();

    if (all_zeros) {
      logging::trace("Removing zeros output to scatter_add: {}",
                     nodeToString(output->node()));
      markNodeForDeletion(output->node());
      return createScatterreduce(graph, {src, index}, axissize, axis,
                                 enable_index_broadcast, sum_reduce);
    }
  }

  return createScatterreduce(graph, {src, index, output}, axissize, axis,
                             enable_index_broadcast, sum_reduce);
}

torch::jit::Node *
meanScatterReduceHandler(torch::jit::Graph *graph, torch::jit::Value *self,
                         torch::jit::Value *index, torch::jit::Value *src,
                         const std::int64_t axis, const std::int64_t axissize,
                         const bool include_self,
                         const bool enable_index_broadcast) {
  static constexpr int32_t sum_reduce =
      static_cast<std::int32_t>(ScatterReduction::Sum);
  auto *ones_self = createConstantFloat32(graph, {1.0}, shapeFromTensor(self));
  auto *ones_src = createConstantFloat32(graph, {1.0}, shapeFromTensor(src));
  torch::jit::Node *count;
  if (include_self) {
    // Count the number of elements reduced to each index.
    count = createScatterreduce(
        graph, {ones_src->output(), index, ones_self->output()}, axissize, axis,
        enable_index_broadcast, sum_reduce);
  } else {
    static constexpr int32_t none_reduce =
        static_cast<std::int32_t>(ScatterReduction::None);
    auto *zeros_src = createConstantFloat32(graph, {0.0}, shapeFromTensor(src));

    // Tensor with zeros where the indices are updated and ones otherwise.
    auto *count_mask = createScatterreduce(
        graph, {zeros_src->output(), index, ones_self->output()}, axissize,
        axis, enable_index_broadcast, none_reduce);

    // Count the number of elements reduced to each index.
    count = createScatterreduce(
        graph, {ones_src->output(), index, count_mask->output()}, axissize,
        axis, enable_index_broadcast, sum_reduce);

    // Put zeros in those indices in self tensor that are not updated,
    // so that they don't impact the reduction result (include_self=False).
    auto *masked_self =
        createScatterreduce(graph, {zeros_src->output(), index, self}, axissize,
                            axis, enable_index_broadcast, none_reduce);
    self = masked_self->output();
  }

  // Sum reduction and then division to calculate `mean`.
  auto *sr = createScatterreduce(graph, {src, index, self}, axissize, axis,
                                 enable_index_broadcast, sum_reduce);
  return createDiv(graph, {sr->output(), count->output()});
}

torch::jit::Node *scatterReduce(torch::jit::Graph *graph,
                                torch::jit::Node *node,
                                const bool enable_index_broadcast) {
  // Signature for scatter_reduce
  // (Tensor src, int dim, Tensor index, Tensor src, string reduce,
  //  bool include_self)
  auto *self = node->input(0);
  auto *dim = node->input(1);
  auto *index = node->input(2);
  auto *src = node->input(3);
  const auto reduce = getReductionMethod(node->input(4)->node());
  const bool include_self = constantToBool(node->input(5)->node());
  const auto src_type = src->type()->expect<c10::TensorType>();
  const auto axis = handleDimensionParam(dim, src_type);
  const auto outshape = shapeFromTensor(node->output(0));
  const auto axissize = outshape.at(axis);

  if (reduce == static_cast<std::int32_t>(ScatterReduction::Mean)) {
    // `Mean` is decomposed as two scatter_reduce sums.
    return meanScatterReduceHandler(graph, self, index, src, axis, axissize,
                                    include_self, enable_index_broadcast);
  }

  if (!include_self) {
    // Mask those indices in `self` that are specified by `index`
    auto *init = createConstantFloat32(graph, {getReductionInitValue(reduce)},
                                       shapeFromTensor(src));
    static constexpr std::int32_t none_reduce =
        static_cast<std::int32_t>(ScatterReduction::None);
    auto *masked_self =
        createScatterreduce(graph, {init->output(), index, self}, axissize,
                            axis, enable_index_broadcast, none_reduce);
    return createScatterreduce(graph, {src, index, masked_self->output()},
                               axissize, axis, enable_index_broadcast, reduce);
  }

  return createScatterreduce(graph, {src, index, self}, axissize, axis,
                             enable_index_broadcast, reduce);
}

torch::jit::Node *scatterReduceHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  const bool enable_index_broadcast = isIndexBroadcastEnabled(node);
  return scatterReduce(graph, node, enable_index_broadcast);
}

torch::jit::Node *indexReduceHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  static constexpr bool enable_index_broadcast = true;
  return scatterReduce(graph, node, enable_index_broadcast);
}

torch::jit::Node *weightNormHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::_weight_norm(Tensor v, Tensor g, int dim) -> Tensor
  auto *v = node->input(0);
  auto *g = node->input(1);
  const auto shape = shapeFromTensor(v);
  auto dim = constantToLong(node->input(2)->node());
  // Correct negative indices
  // PyTorch handles dim -1 in a special way - it computes the
  // norm over all dimensions. We handle that case separately
  if (dim < -1) {
    dim += shape.size();
  }
  std::vector<std::int64_t> axes(shape.size());
  std::iota(axes.begin(), axes.end(), 0);

  // If we have the special case dim -1: We don't erase any
  // axes so that the norm is computed over all dimensions)
  if (dim != -1) {
    axes.erase(axes.begin() + dim);
  }

  std::vector<torch::jit::Value *> axes_constants;
  axes_constants.reserve(axes.size());
  for (auto d : axes) {
    axes_constants.push_back(wrapInConstant1D(graph, d));
  }
  // tensorNormHandler expects ListConstruct for axes_constants
  torch::jit::Value *axes_list =
      createAndInsertNode(graph, c10::prim::ListConstruct, axes_constants)
          ->output();
  // Order 2 norm
  auto *p = wrapInConstant1D(graph, 2);
  // Keep the normalised dims to enable broadcasting
  auto *keepdim = wrapInConstant1D(graph, 1);

  // tensorNormHandler
  auto norm_handler = getHandler(c10::aten::norm);
  // PyTorch defines the weight calculation as
  //   w = g * v / norm(v)
  // This can be rewritten as
  //   w = v * g / norm(v)
  // Which is slightly more efficient, since it doesn't require
  // expanding g to be broadcastable with v
  auto *norm_v =
      createHandlerOperation(graph, norm_handler, {v, p, axes_list, keepdim});
  auto *scaled_v = createDiv(graph, {g, norm_v->output()});

  return createMul(graph, {v, scaled_v->output()});
}

torch::jit::Node *setAvailableMemoryHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  // poptorch::set_available_memory(Tensor, float) -> Tensor
  auto *x = node->input(0);
  auto *y = node->input(1);
  const auto t0 = constantToFloat(y->node());
  return createSetAvailableMemory(graph, x, t0);
}

torch::jit::Node *randintHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto *out = node->output(0);
  const auto shape = shapeFromTensor(out);
  const auto scalar_type = getNodeScalarType(out);
  // Note: the popart range is closed whereas the pytorch range is expected to
  // be half open
  const auto high = constantToFloat(node->input(1)->node()) - 1.0f;
  const auto low = constantToFloat(node->input(0)->node());
  auto *ints =
      createRandomUniform(graph, out, shape, high, low, c10::ScalarType::Int);
  return createCast(graph, ints->output(0), scalar_type);
}

torch::jit::Node *randomHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto *out = node->input(0);
  const auto shape = shapeFromTensor(out);
  const auto scalar_type = getNodeScalarType(out);
  // Note: the popart range is closed whereas the pytorch range is expected to
  // be half open
  const auto high = constantToFloat(node->input(2)->node()) - 1.0f;
  const auto low = constantToFloat(node->input(1)->node());
  auto *ints =
      createRandomUniform(graph, out, shape, high, low, c10::ScalarType::Int);
  return createCast(graph, ints->output(0), scalar_type);
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::einsum, einsumHandler);
  registerHandler(c10::aten::meshgrid, meshgridHandler);
  registerHandler(c10::aten::cartesian_prod, cartesianProdHandler);
  registerHandler(c10::aten::tensordot, tensordotHandler);
  registerHandler(c10::aten::scatter_add, scatterAddHandler);
  registerHandler(c10::aten::scatter_reduce, scatterReduceHandler);
  registerHandler(c10::aten::index_reduce, indexReduceHandler);
  registerHandler(c10::aten::_weight_norm, weightNormHandler);
  registerHandler(c10::aten::randint, randintHandler);
  registerHandler(c10::aten::random_, randomHandler);
  registerHandler(symbols::poptorch::set_available_memory,
                  setAvailableMemoryHandler);
}

} // namespace poptorch
