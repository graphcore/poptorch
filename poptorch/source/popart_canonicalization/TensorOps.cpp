// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace {
torch::jit::Node *sizeHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  //  aten::size(Tensor input, int dim) -> int
  std::vector<std::int64_t> shape = shapeFromTensor(node->input(0));
  std::int64_t dim = constantToLong(node->input(1)->node());
  return createConstantInt(graph, {shape[dim]}, {1});
}

torch::jit::Node *numToTensorHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // Should be a tensor already
  ERROR_ON(node->input(0)->node()->kind() !=
           symbols::poptorch::tensor_constant);
  UNUSED(graph);
  node->output()->replaceAllUsesWith(node->input(0));
  markNodeForDeletion(node);
  return nullptr;
}

torch::jit::Node *flipHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::flip(Tensor self, int[] dims) -> Tensor
  auto *input = node->input(0);
  // Use output shape because input shape might not exist
  // if the input is the result of another operation
  auto input_shape = shapeFromTensor(node->output());
  auto dims = constantToLongVec(node->input(1)->node());
  for (auto &dim : dims) {
    if (dim < 0) {
      dim += input_shape.size();
    }
  }
  return createReverse(graph, {input}, dims);
}

// Input tensor of shape [M, N, ...] is repeated in [R1, R2, ...]
// dimensions by:
//   1) transforming to [1, M, 1, N, ...]
//   2) expanding to [R1, M, R2, N, ...]
//   3) reshaping to [R1*M, R2*N, ...]
torch::jit::Node *repeatHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  torch::jit::Value *input = node->input(0);
  std::vector<std::int64_t> dim_repeats =
      constantToLongVec(node->input(1)->node());
  std::vector<std::int64_t> old_shape = shapeFromTensor(input);
  std::vector<std::int64_t> new_shape = shapeFromTensor(node->output());

  // If repeat dimensions exceed shape dimensions, pad the front of the
  // original shape with singleton dimensions so that it can
  // be expanded

  std::size_t padding = dim_repeats.size() > old_shape.size()
                            ? dim_repeats.size() - old_shape.size()
                            : 0;

  std::vector<std::int64_t> dim_expands;
  std::vector<std::int64_t> transform_shape;

  for (std::size_t i = 0; i < dim_repeats.size(); i++) {
    dim_expands.push_back(dim_repeats[i]);

    std::int64_t padded_dim = i < padding ? 1 : old_shape[i - padding];
    if (padded_dim > 1 && dim_repeats[i] > 1) {
      transform_shape.push_back(1);
      dim_expands.push_back(padded_dim);
    }
    transform_shape.push_back(padded_dim);
  }

  auto *reshape = createReshape(graph, input, transform_shape);
  auto *expand = createExpand(
      graph, {reshape->output(), intVectorToIrConstant(graph, dim_expands)});

  return createReshape(graph, expand->output(), new_shape);
}

torch::jit::Node *rollHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
  auto *input = node->input(0);
  auto input_shape = shapeFromTensor(input);
  auto shifts = constantToLongVec(node->input(1)->node());
  auto dims = constantToLongVec(node->input(2)->node());

  bool reshape_output = false;
  if (dims.empty()) {
    // If dims not provided, a flattened version of the tensor is rolled and
    // then reshaped back.
    ERROR_ON_MSG(shifts.size() != 1,
                 "The 'shifts' argument of the roll op must be a scalar when "
                 "'dims' is not specified.");
    input = createFlatten(graph, {input}, 0)->output();
    int64_t flattened_size = std::accumulate(
        input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
    input_shape.clear();
    input_shape.push_back(1);
    input_shape.push_back(flattened_size);
    dims.push_back(1);
    reshape_output = true;
  } else {
    ERROR_ON_MSG(shifts.size() != dims.size(),
                 "The 'shifts' and 'dims' arguments of the roll op must be the "
                 "same size.");
  }

  torch::jit::Value *output = input;
  auto number_of_dims = input_shape.size();
  for (size_t i = 0; i < dims.size(); ++i) {
    auto current_dim = dims.at(i);
    // Match the torch API of requiring dim in [-len(shape), len(shape)-1]
    ERROR_ON_MSG(((static_cast<size_t>(current_dim) >= number_of_dims) &&
                  (current_dim >= 0)) ||
                     ((static_cast<size_t>(-current_dim) > number_of_dims) &&
                      (current_dim < 0)),
                 "Dimension out of range at index "
                     << i << " (expected to be in range of ["
                     << -static_cast<std::int64_t>(number_of_dims) << ", "
                     << number_of_dims - 1 << "], but got " << current_dim
                     << ") in the roll op.");

    current_dim = (current_dim + number_of_dims) % number_of_dims;

    auto current_dim_size = input_shape.at(current_dim);
    // Handle overreaching and negative shifts.
    auto split = (((-shifts.at(i)) % current_dim_size) + current_dim_size) %
                 current_dim_size;
    auto *chunks = createSplit(graph, {output}, 2, current_dim,
                               {split, current_dim_size - split});
    output =
        createConcat(graph, {chunks->output(1), chunks->output(0)}, current_dim)
            ->output();
  }

  if (reshape_output) {
    return createReshape(graph, output, shapeFromTensor(node->input(0)));
  }
  return output->node();
}

torch::jit::Node *cloneHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor

  // Identity will just create a copy
  return createIdentity(graph, {node->input(0)});
}

torch::jit::Node *copyHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor
  auto *dest = node->input(0);
  auto *src = node->input(1);
  at::ScalarType dest_type = getNodeScalarType(dest);
  at::ScalarType src_type = getNodeScalarType(src);

  torch::jit::Node *copy = nullptr;

  if (src_type == dest_type) {
    copy = createIdentity(graph, {src});
  } else {
    copy = createCast(graph, src, dest_type);
  }
  ERROR_ON(copy == nullptr);

  copy->output()->setType(
      copy->output()->type()->expect<c10::TensorType>()->withRequiresGrad(
          src->type()->expect<c10::TensorType>()->requiresGrad()));

  return copy;
}

torch::jit::Node *justReturnFalse(torch::jit::Graph *graph,
                                  torch::jit::Node * /*unused*/) {
  c10::IValue value{false};
  torch::jit::Value *val = insertConstant(graph, value);
  return val->node();
}

torch::jit::Node *linearHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor
  auto *x = node->input(0);
  auto *w = node->input(1);
  auto *b = node->input(2);

  auto *w_t = createTranspose(graph, {w}, {1, 0});
  auto *output = createMatmul(graph, {x, w_t->output()});

  if (!isNone(b)) {
    output = createAdd(graph, {output->output(), b});
  }
  return output;
}

torch::jit::Node *gatherHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto *input = node->input(0);
  auto tensor_type = input->type()->expect<c10::TensorType>();
  auto axis = handleDimensionParam(node->input(1), tensor_type);
  auto *indices = node->input(2);
  auto input_shape = shapeFromTensor(input);
  auto index_shape = shapeFromTensor(indices);
  auto stride = input_shape[axis];

  for (unsigned s = 0; s < input_shape.size(); ++s) {
    if (s != axis) {
      ERROR_ON(input_shape[s] < index_shape[s]);
    }
  }

  // Move gather axis to the innermost dim
  std::vector<int64_t> permutation;
  unsigned input_num_dims = input_shape.size();
  permutation.resize(input_num_dims);
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.push_back(permutation[axis]);
  permutation.erase(permutation.begin() + axis);

  if (axis != input_num_dims - 1) {
    input = createTranspose(graph, {input}, permutation)->output();
    input_shape.push_back(input_shape[axis]);
    input_shape.erase(input_shape.begin() + axis);
  }
  // Flatten the data
  auto *flatten_input = createFlatten(graph, {input}, 0)->output();
  int64_t num_offsets = std::accumulate(index_shape.begin(), index_shape.end(),
                                        1, std::multiplies<int64_t>());
  num_offsets /= index_shape[axis];

  // Transpose the indices to make them broadcastable with offsets
  std::vector<int64_t> idx_permutation;
  idx_permutation.resize(index_shape.size());
  std::iota(idx_permutation.begin(), idx_permutation.end(), 0);
  idx_permutation.insert(idx_permutation.begin(), idx_permutation[axis]);
  idx_permutation.erase(idx_permutation.begin() + axis + 1);

  if (axis != 0) {
    indices = createTranspose(graph, {indices}, idx_permutation)->output();
    index_shape.insert(index_shape.begin(), index_shape[axis]);
    index_shape.erase(index_shape.begin() + (axis + 1));
  }
  // Create shape for offsets that is broadcastable with indices tensor
  std::vector<int64_t> offset_shape = {index_shape.begin() + 1,
                                       index_shape.end()};
  // Make the offsets
  std::vector<int64_t> offsets_val;
  int64_t num_data = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                     std::multiplies<int64_t>());
  num_data /= input_shape[input_num_dims - 1];
  torch::jit::Value *offsets;

  // Case where one or more indices dims size < data size
  if (num_offsets != num_data) {
    // Create the offsets tensor from data_size
    // then slice it to match indices_size
    auto data_shape = shapeFromTensor(node->input(0));
    data_shape.insert(data_shape.begin(), data_shape[axis]);
    data_shape.erase(data_shape.begin() + (axis + 1));
    std::vector<int64_t> temp_offsets_shape = {data_shape.begin() + 1,
                                               data_shape.end()};
    offsets_val.resize(num_data);
    std::iota(offsets_val.begin(), offsets_val.end(), 0);

    for (auto &v : offsets_val) {
      v *= stride;
    }
    offsets =
        createConstantInt(graph, offsets_val, temp_offsets_shape)->output();

    for (unsigned k = 0; k < offset_shape.size(); ++k) {
      if (offset_shape[k] != temp_offsets_shape[k]) {
        offsets = createSlice(graph, {offsets}, {offset_shape[k]}, {0}, {k})
                      ->output();
      }
    }
  } else {
    offsets_val.resize(num_offsets);
    std::iota(offsets_val.begin(), offsets_val.end(), 0);

    for (auto &v : offsets_val) {
      v *= stride;
    }
    offsets = createConstantInt(graph, offsets_val, {offset_shape})->output();
  }

  auto *new_indices = createAdd(graph, {indices, offsets})->output();
  // Gather the elements
  auto *output = createGather(graph, {flatten_input, new_indices}, 1)->output();
  // remove the dim-0 added by gather
  output = createSqueeze(graph, {output}, {0})->output();
  // transpose back to the original indices shape if needed
  if (axis != 0) {
    std::iota(idx_permutation.begin(), idx_permutation.end(), 0);
    idx_permutation.erase(idx_permutation.begin());
    idx_permutation.insert(idx_permutation.begin() + axis, 0);
    output = createTranspose(graph, {output}, idx_permutation)->output();
  }
  return output->node();
}

torch::jit::Node *scatterHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto *input = node->input(0);
  const auto input_type = input->type()->expect<c10::TensorType>();
  const auto dim = handleDimensionParam(node->input(1), input_type);
  auto *index = node->input(2);
  auto *src = node->input(3);

  // `scatter` can be passed a single value for `src` as a tensor constant, so
  // broadcast it up.
  if (isConstantScalar(src)) {
    auto *shape = intVectorToIrConstant(graph, shapeFromTensor(index));
    src = createExpand(graph, {src, shape})->output();
  }

  ERROR_ON_MSG(node->inputs().size() > 4,
               "Reductions supplied to torch.scatter are currently "
               "unsupported; consider using torch.scatter_add for 'add' "
               "reductions.");

  // scatter(input, index, src, dimension(dim, TensorType(input)))
  return createScatter(graph, {input, index, src}, dim);
}

torch::jit::Node *fullCommon(torch::jit::Graph *graph, torch::jit::Value *v,
                             at::ScalarType type,
                             const std::vector<int64_t> &shape) {
  auto *vn = v->node();
  auto stype = coerceToSupportedType(type);
  if (isTensorConstant(vn) && vn->output()->type()->cast<c10::TensorType>()) {
    auto v_scalar = getNodeTensorAttrValue(vn).to(stype).item();
    return tensorToConstant(graph, at::full(shape, v_scalar, stype));
  }
  auto *v_cast = createCast(graph, v, stype)->output();
  auto *c_shape = intVectorToIrConstant(graph, shape);
  return createExpand(graph, {v_cast, c_shape});
}

torch::jit::Node *fullHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::full(int[] size, Scalar fill_value) -> Tensor
  auto *v = node->input(1);
  auto *shape = node->input(0);
  auto lv_shape = constantToLongVec(shape->node());
  auto type = getNodeScalarType(shape);
  return fullCommon(graph, v, type, lv_shape);
}

torch::jit::Node *fullLikeHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::full_like(Tensor self, Scalar fill_value) -> Tensor
  auto *v = node->input(1);
  auto *like = node->output(0);
  auto like_shape = shapeFromTensor(like);
  auto like_type = getNodeScalarType(like);
  return fullCommon(graph, v, like_type, like_shape);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::size, sizeHandler);
  registerHandler(c10::prim::NumToTensor, numToTensorHandler);
  registerHandler(c10::aten::flip, flipHandler);
  registerHandler(c10::aten::repeat, repeatHandler);
  registerHandler(c10::aten::is_complex, justReturnFalse);
  registerHandler(c10::aten::roll, rollHandler);
  registerHandler(c10::aten::clone, cloneHandler);
  registerHandler(c10::aten::copy_, copyHandler);
  registerHandler(c10::aten::linear, linearHandler);
  registerHandler(c10::aten::gather, gatherHandler);
  registerHandler(c10::aten::scatter, scatterHandler);
  registerHandler(c10::aten::full, fullHandler);
  registerHandler(c10::aten::full_like, fullLikeHandler);
}

} // namespace poptorch
