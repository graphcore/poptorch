// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
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

  auto reshape = createReshape(graph, input, transform_shape);
  auto expand = createExpand(
      graph, {reshape->output(), intVectorToIrConstant(graph, dim_expands)});

  return createReshape(graph, expand->output(), new_shape);
}

torch::jit::Node *rollHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
  auto input = node->input(0);
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
  for (size_t i = 0; i < dims.size(); ++i) {
    auto current_dim = dims.at(i);
    ERROR_ON_MSG(static_cast<size_t>(current_dim) >= input_shape.size() ||
                     current_dim < 0,
                 "Dimension out of range in the roll op.");

    auto current_dim_size = input_shape.at(current_dim);
    // Handle overreaching and negative shifts.
    auto current_shift =
        ((shifts.at(i) % current_dim_size) + current_dim_size) %
        current_dim_size;

    // Duplicate the rolling dimension and then slice based on the shift.
    auto duplicated = createConcat(graph, {output, output}, current_dim);
    auto start = wrapInConstant1D(graph, current_dim_size - current_shift);
    auto end = wrapInConstant1D(graph, 2 * current_dim_size - current_shift);
    auto axis = wrapInConstant1D(graph, current_dim);
    output =
        createSlice(graph, {duplicated->output(), start, end, axis})->output();
  }

  if (reshape_output) {
    return createReshape(graph, output, shapeFromTensor(node->input(0)));
  }
  return output->node();
}

torch::jit::Node *cloneHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor

  // Use cast to same type for cloning
  at::ScalarType dest_type = getNodeScalarType(node->input(0));
  return createCast(graph, node->input(0), dest_type);
}

torch::jit::Node *copyHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor
  at::ScalarType dest_type = getNodeScalarType(node->input(0));

  return createCast(graph, node->input(1), dest_type);
}

torch::jit::Node *linearHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor
  auto x = node->input(0);
  auto w = node->input(1);
  auto b = node->input(2);

  auto w_t = createTranspose(graph, {w}, {1, 0});
  auto output = createMatmul(graph, {x, w_t->output()});

  if (!isNone(b)) {
    output = createAdd(graph, {output->output(), b});
  }
  return output;
}

torch::jit::Node *gatherHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  auto input = node->input(0);
  auto axis = constantToLong(node->input(1)->node());
  auto indices = node->input(2);
  auto scalar_type = getNodeScalarType(input);
  auto input_shape = shapeFromTensor(input);
  auto index_shape = shapeFromTensor(indices);
  ERROR_ON_MSG(input_shape.size() != index_shape.size(),
               "Index and input of mismatching rank!");

  std::vector<int64_t> permutation;
  permutation.resize(input_shape.size());

  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[0], permutation[axis]);

  // transpose the input so that we gather elements along axis 0
  if (axis != 0) {
    input = createTranspose(graph, {input}, permutation)->output();
    indices = createTranspose(graph, {indices}, permutation)->output();
    std::swap(input_shape[0], input_shape[axis]);
    std::swap(index_shape[0], index_shape[axis]);
  }

  auto input_dims = input_shape[0];
  auto index_dims = index_shape[0];

  // after perfoming the transposiztion we know that we are gathering along
  // axis 0. We'll perform gather for each slice along axis 0 - call this slice
  // a plane. The shape of the plane is the shape of the index tensor, less
  // the axis 0 dimension.
  std::vector<std::int64_t> plane_shape(++index_shape.begin(),
                                        index_shape.end());

  bool slice_needed = false;
  for (unsigned i = 1; i < index_shape.size(); ++i) {
    if (index_shape[i] < input_shape[i]) {
      slice_needed = true;
    }
  }

  // slice input planes to plane_shape
  if (slice_needed) {
    std::vector<std::int64_t> start(plane_shape.size(), 0);
    std::vector<std::int64_t> axes;

    axes.resize(input_shape.size() - 1);
    std::iota(axes.begin(), axes.end(), 1);

    auto start_vals =
        createConstantInt(graph, start,
                          {static_cast<std::int64_t>(start.size())})
            ->output();
    auto end_vals =
        createConstantInt(graph, plane_shape,
                          {static_cast<std::int64_t>(plane_shape.size())})
            ->output();
    auto axes_vals =
        createConstantInt(graph, axes, {static_cast<std::int64_t>(axes.size())})
            ->output();
    input =
        createSlice(graph, {input, start_vals, end_vals, axes_vals})->output();
  }

  std::vector<std::int64_t> dims_vector;
  dims_vector.resize(input_dims);
  std::iota(dims_vector.begin(), dims_vector.end(), 0);

  std::vector<torch::jit::Value *> plane_mask;
  std::transform(dims_vector.begin(), dims_vector.end(),
                 std::back_inserter(plane_mask),
                 [&](std::int64_t i) -> torch::jit::Value * {
                   return createConstantInt(graph, {i}, plane_shape)->output();
                 });

  std::vector<std::int64_t> input_slice_size(input_dims, 1);
  std::vector<std::int64_t> index_slice_size(index_dims, 1);
  auto input_split =
      createSplit(graph, {input}, input_dims, 0, input_slice_size);
  auto index_split =
      createSplit(graph, {indices}, index_dims, 0, index_slice_size);

  std::vector<torch::jit::Value *> planes;
  for (int i = 0; i < index_dims; ++i) {
    auto plane =
        createConstantFloatLike(graph, input, {0.0}, plane_shape)->output();

    for (int j = 0; j < input_dims; ++j) {
      auto eq =
          createEqual(graph, {index_split->output(i), plane_mask[j]})->output();
      auto mask = createCast(graph, eq, scalar_type)->output();
      auto mul = createMul(graph, {input_split->output(j), mask})->output();
      plane = createAdd(graph, {plane, mul})->output();
    }

    planes.push_back(plane);
  }

  auto result = createConcat(graph, planes, 0)->output();

  // transpose the result
  if (axis != 0) {
    result = createTranspose(graph, {result}, permutation)->output();
  }

  return result->node();
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::size, sizeHandler);
  registerHandler(c10::prim::NumToTensor, numToTensorHandler);
  registerHandler(c10::aten::repeat, repeatHandler);
  registerHandler(c10::aten::roll, rollHandler);
  registerHandler(c10::aten::clone, cloneHandler);
  registerHandler(c10::aten::copy_, copyHandler);
  registerHandler(c10::aten::linear, linearHandler);
  registerHandler(c10::aten::gather, gatherHandler);
}

} // namespace poptorch
