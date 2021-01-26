// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"

namespace poptorch {
namespace {

torch::jit::Node *indexHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::index(Tensor self, Tensor?[] indices)
  torch::jit::Value *x = node->input(0);
  std::vector<torch::jit::Value *> indices =
      handleTensorList(node->input(1)->node());
  auto shape = shapeFromTensor(x);
  if (indices.size() > shape.size()) {
    std::stringstream ss;
    ss << "too many indices for tensor of dimension " << shape.size()
       << " (got " << indices.size() << ")";
    ERROR(ss.str());
  }

  std::size_t pad = 0;
  std::size_t index_size = 0;
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
      if (s.size() > index_size) {
        index_size = s.size();
      }
      indexed = true;
    }
  }

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
      nones_indexed++;

      indices[i] =
          createReshape(graph, intVectorToIrConstant(graph, idx), new_shape)
              ->output();
    } else {
      const auto index_shape = shapeFromTensor(indices[i]);
      std::vector<std::int64_t> new_shape;

      auto ones_generator = []() { return 1; };
      if (pad_front) {
        std::generate_n(std::back_inserter(new_shape), pad, ones_generator);
      }
      std::copy(index_shape.begin(), index_shape.end(),
                std::back_inserter(new_shape));
      if (!pad_front) {
        std::generate_n(std::back_inserter(new_shape), pad, ones_generator);
      }
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

  std::vector<std::int64_t> gather_shape = {-1};
  std::copy_n(shape.begin() + indices.size(), shape.size() - indices.size(),
              std::back_inserter(gather_shape));
  // Flatten the tensor being indexed into [-1, u1, u2, ..., uN] where
  // each u is a dimension not being indexed into
  auto flatten = createReshape(graph, x, gather_shape);
  // Gather in first dimension using calculated indices into partially flattened
  // tensor
  return createGather(graph, {flatten->output(), flat_indices->output()}, 0);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::index, indexHandler);
}

} // namespace poptorch
