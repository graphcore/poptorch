// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <stack>

#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {

namespace {
void recursivelySwitchType(torch::jit::Node *node,
                           const torch::jit::TypePtr &new_type) {
  for (auto use : node->output()->uses()) {
    ERROR_ON(use.user->kind() == c10::prim::ListConstruct);

    // No known JIT model causes this, but one may emerge in which case
    // this algorithm will need to handle it.
    ERROR_ON(use.user->kind() == c10::prim::TupleUnpack);

    if (use.user->kind() == c10::prim::TupleConstruct) {
      const auto &tuple_elements =
          use.user->output()->type()->expect<c10::TupleType>()->elements();

      std::vector<c10::TypePtr> new_types;
      new_types.reserve(tuple_elements.size());
      std::copy(tuple_elements.begin(), tuple_elements.end(),
                std::back_inserter(new_types));

      // This will be the list or nested tuple containing list
      new_types[use.offset] = new_type;

      auto new_tuple_type = c10::TupleType::create(new_types);

      use.user->output()->setType(new_tuple_type);
      recursivelySwitchType(use.user, new_tuple_type);
    }
  }
}

} // namespace

void addListNumElements(torch::jit::Graph *graph, bool revert) {
  for (torch::jit::Node *node : graph->nodes()) {
    logging::LogContext ctx("addListNumElements processing " +
                            nodeToString(node));

    if (node->kind() == c10::prim::ListConstruct) {
      auto list_inputs = node->inputs();

      // Lists should never be nested as the JIT tracer does not support,
      // but always good to check in case.
      for (auto input : list_inputs) {
        ERROR_ON(input->type()->kind() == c10::TypeKind::ListType);
      }

      c10::TypePtr new_type;
      if (revert) {
        // Revert back to the orgiinal type
        auto lot_type =
            node->output()->type()->expect<ListTypeWithNumElements>();
        new_type = lot_type->getOriginalListType();
      } else {
        // Switch to a ListTypeWithNumElements
        auto orig_type = node->output()->type()->expect<c10::ListType>();
        auto num_elements = list_inputs.size();
        new_type = std::make_shared<ListTypeWithNumElements>(
            orig_type->getElementType(), num_elements);
      }

      node->output()->setType(new_type);

      // Any tuples which have te list need fixing.
      recursivelySwitchType(node, new_type);
    }
  }
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
