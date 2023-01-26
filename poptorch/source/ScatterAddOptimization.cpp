// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>
#include <vector>

#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {

void removeScatterAddIndexExpansion(torch::jit::Graph *graph) {
  const logging::LogContext ctx{"ScatterAddOptimization"};

  std::vector<torch::jit::Node *> to_delete;

  for (auto *node : graph->nodes()) {
    if (node->kind() != c10::aten::scatter_add &&
        node->kind() != c10::aten::scatter_add_ &&
        node->kind() != c10::aten::scatter_reduce &&
        node->kind() != c10::aten::scatter_reduce_) {
      continue;
    }

    // aten::scatter_add(Tensor self, int dim, Tensor index,
    //                   Tensor src) -> Tensor
    // aten::scatter_add_(Tensor(a!) self, int dim, Tensor index,
    //                    Tensor src) -> Tensor(a!)
    // aten::scatter_reduce(Tensor self, int dim, Tensor index,
    //                      Tensor src, str reduce, bool include_self) -> Tensor
    // aten::scatter_reduce_(Tensor(a!) self, int dim, Tensor index,
    //                       Tensor src, str reduce, bool include_self)
    //                      -> Tensor(a!)
    auto *index = node->input(2);
    auto *index_producer = index->node();

    // Only remove index expansions.
    if (index_producer->kind() != c10::aten::expand &&
        index_producer->kind() != c10::aten::expand_as) {
      continue;
    }

    // aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor
    // aten::expand_as(Tensor self, Tensor other) -> Tensor
    auto *src = node->input(3);
    auto *original_index = index_producer->input(0);
    auto expanded_index_shape = shapeFromTensor(index);

    // Make sure removal is valid
    if (index->uses().size() > 1 ||
        shapeFromTensor(src) != expanded_index_shape) {
      continue;
    }

    logging::trace("Removing index expansion node: {}",
                   nodeToString(index_producer));
    node->replaceInputWith(index, original_index);
    to_delete.push_back(index_producer);
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

} // namespace poptorch
