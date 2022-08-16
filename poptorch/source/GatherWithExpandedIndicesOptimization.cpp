// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_set>

#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"

namespace poptorch {

void simplifyGatherWithExpandedIndices(torch::jit::Graph *graph) {
  logging::LogContext ctx{"GatherWithExpandedIndicesOptimisation"};

  std::unordered_set<torch::jit::Node *> to_delete;

  for (auto *node : graph->nodes()) {
    if (node->kind() != c10::aten::gather) {
      continue;
    }

    // aten::gather(Tensor self, int dim, Tensor index, *, bool
    //              sparse_grad=False) -> Tensor
    auto *input = node->input(0);
    const size_t gather_dim = handleDimensionParam(
        node->input(1), input->type()->expect<c10::TensorType>());
    auto *indices = node->input(2);
    auto *expand_node = indices->node();

    // Only remove index expansions.
    if (expand_node->kind() != c10::aten::expand &&
        expand_node->kind() != c10::aten::expand_as) {
      continue;
    }

    WithNodeMetadata meta(node);
    // aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor
    // aten::expand_as(Tensor self, Tensor other) -> Tensor
    auto *original_indices = expand_node->input(0);
    auto original_indices_shape = shapeFromTensor(original_indices);

    // Get the (single) expanded dimension
    std::vector<int64_t> expand_shape{};
    if (expand_node->kind() == c10::aten::expand) {
      expand_shape = constantToLongVec(expand_node->input(1)->node());
    } else {
      expand_shape = shapeFromTensor(expand_node->input(1));
    }

    std::vector<size_t> expand_dims{};
    for (size_t i = 0; i < expand_shape.size(); i++) {
      if (expand_shape[i] > original_indices_shape[i]) {
        expand_dims.push_back(i);
      }
    }
    if (expand_dims.size() != 1) {
      continue;
    }
    size_t expand_dim = expand_dims[0];

    // Only optimise if:
    // * source tensor's shape has 2 dimensions of length > 1
    // * dimension of gather, and dimension of expand are the 2 dimensions of
    //   length > 1
    const auto self_shape = shapeFromTensor(input);
    std::vector<size_t> non_singleton_dimensions{};
    for (size_t i = 0; i < self_shape.size(); i++) {
      if (self_shape[i] > 1) {
        non_singleton_dimensions.push_back(i);
      }
    }
    if (non_singleton_dimensions.size() != 2) {
      continue;
    }

    const auto ga_it = std::find(non_singleton_dimensions.begin(),
                                 non_singleton_dimensions.end(), gather_dim);
    const auto ex_it = std::find(non_singleton_dimensions.begin(),
                                 non_singleton_dimensions.end(), expand_dim);
    if (ga_it == ex_it || ga_it == non_singleton_dimensions.end() ||
        ex_it == non_singleton_dimensions.end()) {
      continue;
    }

    // Replace the aten::expand -> aten::gather with an
    // aten::squeeze -> aten::index_select
    logging::debug("Optimising gather: {}", nodeToString(node));
    std::vector<int64_t> squeezed_shape;
    std::copy_if(original_indices_shape.begin(), original_indices_shape.end(),
                 std::back_inserter(squeezed_shape),
                 [](auto dim) { return dim > 1; });

    torch::jit::WithInsertPoint insert_point(node);

    torch::jit::Node *squeezed =
        createAndInsertNode(graph, c10::aten::squeeze, {original_indices},
                            ImplicitCast::None, OutputType::AsFirstInput);
    squeezed->output()->setType(
        original_indices->type()->expect<c10::TensorType>()->withSizes(
            squeezed_shape));
    torch::jit::Node *gathered =
        createAndInsertNode(graph, c10::aten::index_select,
                            {input, node->input(1), squeezed->output()},
                            ImplicitCast::None, OutputType::AsFirstInput)
            ->output()
            ->node();

    to_delete.insert(node);
    to_delete.insert(expand_node);

    if (node->hasUses()) {
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        // As well as replacing the use, this will copy across shape/type
        // if not explicitly set.
        replaceOutputUse(node, gathered, i);
      }
    }
  }

  // Remove the dead nodes.
  searchAndPossiblyDestroy(to_delete);
}

} // namespace poptorch
