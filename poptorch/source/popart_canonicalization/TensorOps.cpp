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
  std::vector<std::int64_t> old_shape = shapeFromTensor(node->input(0));
  std::vector<std::int64_t> new_shape = shapeFromTensor(node->output());
  std::vector<std::int64_t> dim_repeats =
      constantToLongVec(node->input(1)->node());
  std::vector<std::int64_t> dim_expands;
  std::vector<std::int64_t> transform_shape;

  // If repeat dimensions exceed shape dimensions, pad the front of the
  // original shape with singleton dimensions so that it can
  // be expanded
  size_t padding = dim_repeats.size() > old_shape.size()
                       ? dim_repeats.size() - old_shape.size()
                       : 0;

  torch::jit::Node *new_node = node->input(0)->node();

  for (std::size_t i = 0; i < dim_repeats.size(); i++) {
    dim_expands.push_back(dim_repeats[i]);

    std::int64_t padded_dim = i < padding ? 1 : old_shape[i - padding];
    if (padded_dim > 1 && dim_repeats[i] > 1) {
      transform_shape.push_back(1);
      dim_expands.push_back(padded_dim);
    }
    transform_shape.push_back(padded_dim);
  }

  new_node = createReshape(graph, new_node->output(), transform_shape);
  new_node = createExpand(
      graph, {new_node->output(), intVectorToIrConstant(graph, dim_expands)});

  return createReshape(graph, new_node->output(), new_shape);
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

  torch::jit::Node *output = input->node();
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
    auto duplicated =
        createConcat(graph, {output->output(), output->output()}, current_dim);
    auto start = wrapInConstant1D(graph, current_dim_size - current_shift);
    auto end = wrapInConstant1D(graph, 2 * current_dim_size - current_shift);
    auto axis = wrapInConstant1D(graph, current_dim);
    output = createSlice(graph, {duplicated->output(), start, end, axis});
  }

  if (reshape_output) {
    return createReshape(graph, output->output(),
                         shapeFromTensor(node->input(0)));
  }
  return output;
}

// NOLINTNEXTLINE
torch::jit::Node *copy_Handler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor
  at::ScalarType dest_type = getNodeScalarType(node->input(0));

  return createCast(graph, node->input(1), dest_type);
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::size, sizeHandler);
  registerHandler(c10::prim::NumToTensor, numToTensorHandler);
  registerHandler(c10::aten::repeat, repeatHandler);
  registerHandler(c10::aten::roll, rollHandler);
  registerHandler(c10::aten::copy_, copy_Handler);
}

} // namespace poptorch
