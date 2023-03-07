// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "ScatterReduction.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
std::int32_t getReductionMethod(const torch::jit::Node *node) {
  const auto kind = node->kind();

  if (kind == torch_scatter::scatter_max) {
    return static_cast<std::int32_t>(ScatterReduction::Max);
  }
  if (kind == torch_scatter::scatter_min) {
    return static_cast<std::int32_t>(ScatterReduction::Min);
  }
  if (kind == torch_scatter::scatter_mul) {
    return static_cast<std::int32_t>(ScatterReduction::Mul);
  }

  ERROR("Unsupported reduction for node: " << nodeToString(node));
}

torch::jit::Node *torchScatterHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  static constexpr bool enable_index_broadcast = true;

  // Signatures for scatter_max, scatter_min, scatter_mul:
  // (Tensor src, Tensor index, int dim, Tensor? out, int? dim_size)
  auto *src = node->input(0);
  auto *index = node->input(1);
  const auto src_type = src->type()->expect<c10::TensorType>();
  const auto axis = handleDimensionParam(node->input(2), src_type);

  auto *opt_out = node->input(3);

  std::vector<torch::jit::Value *> args{src, index};
  if (!isNone(opt_out)) {
    args.push_back(opt_out);
  }

  auto shape = shapeFromTensor(node->output(0));
  auto axis_size = shape.at(axis);

  auto *opt_axis_size = node->input(4);
  if (!isNone(opt_axis_size)) {
    axis_size = constantToInt(opt_axis_size->node());
  }

  auto *result =
      createScatterreduce(graph, args, axis_size, axis, enable_index_broadcast,
                          getReductionMethod(node));

  if (node->outputs().size() == 1) {
    return result;
  }

  // Both scatter_max and scatter_min return two outputs where the second one
  // is the index but most often this second output is simply ignored.
  if (!node->output(1)->hasUses()) {
    // the indices output is unused so is safe to delete
    node->eraseOutput(1);
    return result;
  }

  // Calculate the indices of the max/min
  const auto ishape = shapeFromTensor(src);
  std::vector<int64_t> index_range_shape(ishape.size(), 1);
  index_range_shape[axis] = ishape[axis];

  const auto gather_handler = getHandler(c10::aten::gather);
  result->output()->setType(src_type->withSizes(shape));
  auto *gather =
      createHandlerOperation(graph, gather_handler,
                             {result->output(), node->input(2), index})
          ->output();

  // true if the scatter chose this location in src, false if we didn't
  auto *mask = createEqual(graph, {gather, src})->output();
  std::vector<std::int64_t> vals(ishape[axis]);
  std::iota(std::begin(vals), std::end(vals), 1);
  auto *index_range =
      createConstantInt(graph, vals, index_range_shape)->output();
  auto *not_chosen =
      createConstantInt(graph, {ishape[axis] + 1}, {1})->output();
  // The 1-based index in src if this location was chosen, ishape[axis] + 1 if
  // it wasn't
  auto *index_of_result =
      createWhere(graph, {mask, index_range, not_chosen})->output();
  // Apply the same scattering to our index tensor as we did to the input tensor
  static constexpr std::int32_t min_reduce =
      static_cast<std::int32_t>(ScatterReduction::Min);
  auto *arg_scatter =
      createScatterreduce(graph, {index_of_result, index}, axis_size, axis,
                          enable_index_broadcast, min_reduce)
          ->output();
  // Now we've got a tensor of 1-based indices, with zeroes where no index
  // was scattered. We need to transform this to zero-based indices, with
  // ishape[axis] where no index was scattered.
  auto *one = createConstantInt(graph, {1}, {1})->output();
  arg_scatter = createSub(graph, {arg_scatter, one})->output();
  arg_scatter = createRemainder(graph, {arg_scatter, not_chosen})->output();

  replaceOutputUse(node->output(0), result->output());
  replaceOutputUse(node->output(1), arg_scatter);
  markNodeForDeletion(node);
  return result;
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(torch_scatter::scatter_max, torchScatterHandler);
  registerHandler(torch_scatter::scatter_min, torchScatterHandler);
  registerHandler(torch_scatter::scatter_mul, torchScatterHandler);
}

} // namespace poptorch
