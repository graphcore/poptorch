// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {
namespace {

struct IndexInfo {
  torch::jit::Value *x_partial_flat;
  torch::jit::Value *indices_partial_flat;
};

std::vector<std::int64_t> padShape(const std::vector<std::int64_t> &shape,
                                   std::size_t pad, bool pad_front) {
  std::vector<std::int64_t> output_shape;
  auto ones_generator = []() { return 1; };
  if (pad_front) {
    std::generate_n(std::back_inserter(output_shape), pad, ones_generator);
  }
  std::copy(shape.begin(), shape.end(), std::back_inserter(output_shape));
  if (!pad_front) {
    std::generate_n(std::back_inserter(output_shape), pad, ones_generator);
  }
  return output_shape;
}

IndexInfo processIndex(torch::jit::Graph *graph, torch::jit::Value *x,
                       std::vector<torch::jit::Value *> *p_indices) {
  auto &indices = *p_indices;
  auto shape = shapeFromTensor(x);

  std::size_t pad = 0;
  std::vector<std::int64_t> index_shape;
  bool indexed = false;
  bool pad_front = true;
  // Calculate the final index size with which the gather operation will be
  // performed
  for (torch::jit::Value *index : indices) {
    if (isNone(index)) {
      if (indexed) {
        pad_front = false;
      }
      pad++;
    } else {
      if (getNodeScalarType(index) == c10::ScalarType::Bool) {
        ERROR("Indexing using boolean/byte tensor masks is unsupported.");
      }

      auto s = shapeFromTensor(index);
      if (s.size() > index_shape.size()) {
        index_shape = s;
      }
      indexed = true;
    }
  }
  std::size_t index_size = index_shape.size();
  std::vector<std::int64_t> flat_indices_shape =
      padShape(index_shape, pad, pad_front);

  std::size_t nones_indexed = 0;
  // Reshape each tensor into shape broadcastable with final output shape
  for (std::size_t i = 0; i < indices.size(); i++) {
    if (isNone(indices[i])) {
      // Optional tensors: 'None' means indexing over entire dimension
      // Replace each None tensor with its explicit index representation
      std::vector<std::int64_t> idx(shape[i]);
      std::iota(idx.begin(), idx.end(), 0);

      std::vector<std::int64_t> new_shape(index_size + pad, 1);
      auto final_shape_index =
          pad_front ? nones_indexed : index_size + nones_indexed;
      new_shape[final_shape_index] = shape[i];
      flat_indices_shape[final_shape_index] = shape[i];
      nones_indexed++;

      indices[i] =
          createReshape(graph, intVectorToIrConstant(graph, idx), new_shape)
              ->output();
    } else {
      const auto original_shape = shapeFromTensor(indices[i]);
      std::vector<std::int64_t> new_shape =
          padShape(original_shape, pad, pad_front);

      indices[i] = createReshape(graph, indices[i], new_shape)->output();
    }
  }

  auto flat_indices = createConstantInt(graph, {0}, {});
  std::int64_t stride = 1;
  // Calculate indices within partially flattened shape
  // Tensors are automatically broadcast to the correct shape during calculation
  for (std::size_t i = 0; i < indices.size(); i++) {
    auto offset = indices[indices.size() - i - 1];
    if (i != 0) {
      offset =
          createMul(graph, {offset, wrapInConstant1D(graph, stride)})->output();
    }
    flat_indices = createAdd(graph, {flat_indices->output(), offset});
    stride *= shape[indices.size() - i - 1];
  }
  // Retain the shape for downstream calculation
  flat_indices =
      createReshape(graph, flat_indices->output(), flat_indices_shape);

  std::vector<std::int64_t> flatten_shape = {-1};
  std::copy_n(shape.begin() + indices.size(), shape.size() - indices.size(),
              std::back_inserter(flatten_shape));
  // Flatten the tensor being indexed into [-1, u1, u2, ..., uN] where
  // each u is a dimension not being indexed into
  auto flatten = createReshape(graph, x, flatten_shape);

  return {flatten->output(), flat_indices->output()};
}

torch::jit::Node *indexHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::index(Tensor self, Tensor?[] indices)
  torch::jit::Value *x = node->input(0);
  std::vector<torch::jit::Value *> indices =
      handleTensorList(node->input(1)->node());

  IndexInfo info = processIndex(graph, x, &indices);
  // Gather in first dimension using calculated indices into partially flattened
  // tensor
  return createGather(graph, {info.x_partial_flat, info.indices_partial_flat},
                      0);
}

torch::jit::Node *indexPutHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::index_put_(Tensor self, Tensor?[] indices, Tensor value, bool
  //                  accumulate)
  torch::jit::Value *x = node->input(0);
  std::vector<torch::jit::Value *> indices =
      handleTensorList(node->input(1)->node());
  torch::jit::Value *v = node->input(2);

  auto fn_gen_none = [graph]() {
    torch::jit::Value *none = graph->create(c10::prim::Constant)->output();
    none->setType(c10::NoneType::get());
    return none;
  };
  auto shape = shapeFromTensor(x);
  // Scatter cannot assign entire dimensions, only individual elements, so we
  // must pad the end of indices with NoneTypes so that the entire input is
  // flattened during indexing
  std::generate_n(std::back_inserter(indices), shape.size() - indices.size(),
                  fn_gen_none);

  IndexInfo info = processIndex(graph, x, &indices);

  auto v_shape = shapeFromTensor(v);
  auto indices_shape = shapeFromTensor(info.indices_partial_flat);
  auto indices_size =
      std::accumulate(indices_shape.begin(), indices_shape.end(), 1,
                      std::multiplies<std::int64_t>{});

  // Ensure value tensor can be broadcast with indexing result
  if (v_shape.size() < indices_shape.size()) {
    v = createReshape(graph, v, {1, -1})->output();
    auto v_size = std::accumulate(v_shape.begin(), v_shape.end(), 1,
                                  std::multiplies<std::int64_t>{});
    // Repeat v to match indices shape
    v = createExpand(graph, {v, intVectorToIrConstant(
                                    graph, {indices_size / v_size, 1})})
            ->output();
  }
  info.indices_partial_flat =
      createReshape(graph, info.indices_partial_flat, {indices_size})->output();
  v = createReshape(graph, v, {indices_size})->output();

  // Scatter in first dimension using calculated indices into fully flattened
  // tensor
  auto scatter = createScatter(
      graph, {info.x_partial_flat, info.indices_partial_flat, v}, 0);
  // Restore original input shape
  return createReshape(graph, scatter->output(), shape);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::index, indexHandler);
  registerHandler(c10::aten::index_put_, indexPutHandler);
}

} // namespace poptorch
