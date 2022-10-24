// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

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
      auto index_dtype = getNodeScalarType(index);
      ERROR_ON_MSG(
          index_dtype == c10::ScalarType::Bool ||
              index_dtype == c10::ScalarType::Byte,
          "Indexing using boolean or byte tensor masks is unsupported because "
          "it would produce dynamic output shapes based on the mask values. "
          "The IPU cannot support dynamic output shapes.");

      auto s = shapeFromTensor(index);
      if (s.size() > index_shape.size()) {
        index_shape = s;
      }
      indexed = true;
    }
  }
  std::size_t const index_size = index_shape.size();
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
      const std::vector<std::int64_t> new_shape =
          padShape(original_shape, pad, pad_front);

      indices[i] = createReshape(graph, indices[i], new_shape)->output();
    }
  }

  auto *flat_indices = indices[indices.size() - 1];
  std::int64_t stride = shape[indices.size() - 1];
  // Calculate indices within partially flattened shape
  // Tensors are automatically broadcast to the correct shape during calculation
  for (auto i = 1u; i < indices.size(); i++) {
    auto *index = indices[indices.size() - i - 1];
    auto *offset =
        createMul(graph, {index, wrapInConstant1D(graph, stride)})->output();
    flat_indices = createAdd(graph, {flat_indices, offset})->output();
    stride *= shape[indices.size() - i - 1];
  }
  // Retain the shape for downstream calculation
  flat_indices =
      createReshape(graph, flat_indices, flat_indices_shape)->output();

  std::vector<std::int64_t> flatten_shape = {-1};
  std::copy_n(shape.begin() + indices.size(), shape.size() - indices.size(),
              std::back_inserter(flatten_shape));
  // Flatten the tensor being indexed into [-1, u1, u2, ..., uN] where
  // each u is a dimension not being indexed into
  auto *flatten = createReshape(graph, x, flatten_shape);

  return {flatten->output(), flat_indices};
}

torch::jit::Node *indexHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::index(Tensor self, Tensor?[] indices)
  torch::jit::Value *x = node->input(0);
  std::vector<torch::jit::Value *> indices =
      handleTensorList(node->input(1)->node());

  const IndexInfo info = processIndex(graph, x, &indices);
  // Gather in first dimension using calculated indices into partially flattened
  // tensor
  return createGather(graph, {info.x_partial_flat, info.indices_partial_flat},
                      0);
}

bool isMaskedAssign(torch::jit::Graph *graph, torch::jit::Value *x,
                    std::vector<torch::jit::Value *> &indices) {
  // Masked fill only takes one index tensor which is broadcastable
  // with the input
  if (indices.size() != 1) {
    return false;
  }

  auto index = indices[0]->type()->expect<c10::TensorType>();
  ERROR_ON(!index->scalarType().has_value());
  auto dtype = index->scalarType().value();

  // Masks must be of type bool or byte
  if (dtype != c10::ScalarType::Bool && dtype != c10::ScalarType::Byte) {
    return false;
  }

  auto mask_shape = shapeFromTensor(indices[0]);
  auto x_shape = shapeFromTensor(x);

  // popart::where expects a bool tensor mask so cast if necessary
  if (dtype == c10::ScalarType::Byte) {
    indices[0] = createCast(graph, indices[0], c10::ScalarType::Bool)->output();
  }

  // Pad indices to enable broadcasting
  if (mask_shape.size() < x_shape.size()) {
    mask_shape.resize(x_shape.size(), 1);
    indices[0] = createReshape(graph, indices[0], mask_shape)->output();
  }
  return true;
}

std::optional<std::int32_t>
canVectorizeInDim(std::vector<torch::jit::Value *> &indices) {
  std::optional<std::int32_t> dim;
  std::int32_t const num_indices = static_cast<std::int32_t>(indices.size());

  for (std::int32_t i = 0; i < num_indices; i++) {
    if (isNone(indices[i])) {
      continue;
    }

    if (dim) {
      // Already found a valid dim but additional indices are specified so
      // cannot vectorise this case.
      return std::nullopt;
    }

    auto idx = indices[i]->type()->expect<c10::TensorType>();
    ERROR_ON(!idx->scalarType().has_value());
    auto dtype = idx->scalarType().value();

    if (!isIntegralType(dtype, false)) {
      return std::nullopt;
    }

    if (idx->dim() != 1 || idx->numel() == 1) {
      return std::nullopt;
    }

    dim = i;
  }

  return dim;
}

void applyInplaceSlice(torch::jit::Node *node, torch::jit::Node *out) {
  // If we're performing an index_put on a slice - this should operate
  // "in-place"
  //
  // Slices are tensor views in torch, and index_put_ should modify the tensor
  // being sliced. To simulate in-place modification to slices, we replace all
  // uses of the tensor being sliced with the output of this operation
  torch::jit::Value *x = node->input(0);

  if (x->node()->kind() == symbols::popart::slice) {
    auto *slice_input = x->node()->input(0);
    // Recursively follow the chain of slices until we find the original tensor
    // actually being sliced
    while (slice_input->node()->kind() == symbols::popart::slice) {
      slice_input = slice_input->node()->input(0);
    }
    slice_input->replaceAllUsesAfterNodeWith(node, out->output());
  }
}

torch::jit::Node *indexPutHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::index_put(Tensor self, Tensor?[] indices, Tensor value, bool
  //                  accumulate)
  torch::jit::Value *x = node->input(0);
  std::vector<torch::jit::Value *> indices =
      handleTensorList(node->input(1)->node());
  torch::jit::Value *v = node->input(2);

  if (isMaskedAssign(graph, x, indices)) {
    return createWhere(graph, {indices[0], v, x});
  }

  auto fn_gen_none = [graph]() {
    torch::jit::Value *none = graph->create(c10::prim::Constant)->output();
    none->setType(c10::NoneType::get());
    return none;
  };
  auto shape = shapeFromTensor(x);
  auto vectorized_dim = canVectorizeInDim(indices);
  auto v_shape = shapeFromTensor(v);
  if (vectorized_dim) {
    logging::trace(
        "Using vectorized ScatterReduce with none reduction in dim {}",
        *vectorized_dim);
    // Expand the value tensor to match the input if necessary
    if (v_shape.size() < shape.size()) {
      auto new_shape = shape;
      // In the vectorised case, the index will always be a 1D tensor
      new_shape[*vectorized_dim] = shapeFromTensor(indices[*vectorized_dim])[0];
      v = createExpand(graph, {v, intVectorToIrConstant(graph, new_shape)})
              ->output();
    }
    auto *out = createScatterreduce(graph, {v, indices[*vectorized_dim], x},
                                    shape[0], *vectorized_dim, 3);
    applyInplaceSlice(node, out);
    return out;
  }

  // ONNX Scatter cannot assign entire dimensions, only individual elements, so
  // we must pad the end of indices with NoneTypes so that the entire input is
  // flattened during indexing
  std::generate_n(std::back_inserter(indices), shape.size() - indices.size(),
                  fn_gen_none);

  IndexInfo info = processIndex(graph, x, &indices);

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
  auto *scatter = createScatter(
      graph, {info.x_partial_flat, info.indices_partial_flat, v}, 0);
  // Restore original input shape
  auto *out = createReshape(graph, scatter->output(), shape);
  applyInplaceSlice(node, out);
  return out;
}

torch::jit::Node *indexFillHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar
  // value) -> Tensor aten::index_fill.int_Tensor(Tensor self, int dim, Tensor
  // index, Tensor value) -> Tensor
  auto *self = node->input(0);
  auto dim = constantToLong(node->input(1)->node());
  auto *index = node->input(2);
  auto *value = node->input(3);
  auto self_dtype = getNodeScalarType(self);
  if (getNodeScalarType(value) != self_dtype) {
    value = createCast(graph, value, self_dtype)->output();
  }

  // Create Tensor?[] indices, where indices[dim] = index, and indices[d] =
  // None, where d < dim
  std::vector<torch::jit::Value *> indices;
  auto fn_gen_none = [graph]() {
    auto *none = graph->createNone();
    insertNodeInGraph(graph, none);
    return none->output();
  };
  std::generate_n(std::back_inserter(indices), dim, fn_gen_none);
  indices.push_back(index);
  auto *list = createAndInsertNode(graph, c10::prim::ListConstruct, indices);
  auto *accumulate = createConstantInt(graph, {0}, {});

  // Re-use index_put handler
  auto index_put_handler = getHandler(c10::aten::index_put);
  return createHandlerOperation(
      graph, index_put_handler,
      {self, list->output(), value, accumulate->output()});
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::index, indexHandler);
  registerHandler(c10::aten::index_put, indexPutHandler);
  registerHandler(c10::aten::index_fill, indexFillHandler);
}

} // namespace poptorch
