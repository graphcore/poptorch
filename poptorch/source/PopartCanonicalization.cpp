// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {

class CanonicalizeImpl {
public:
  static void run(torch::jit::Graph *graph);
};

/*
 * ConvertAtenToPopart implementation.
 */

void CanonicalizeImpl::run(torch::jit::Graph *graph) {
  for (torch::jit::Node *node : graph->nodes()) {
    logging::LogContext ctx("PopartCanonicalization processing " +
                            nodeToString(node));
    torch::jit::WithInsertPoint insert_point(node);
    torch::jit::Node *new_node = nullptr;
    torch::jit::Symbol kind = node->kind();

    if (SymbolHandler handler = getHandler(kind)) {
      new_node = handler(graph, node);
    }

    // If we have a new node add it and replace the old use.
    if (new_node) {
      // Mark this node for deletion.
      markNodeForDeletion(node);
      ERROR_ON(node->outputs().size() != new_node->outputs().size());

      if (node->hasUses()) {
        for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
          replaceOutputUse(node, new_node, i);
        }
      }
    }
  }

  // Build a list of nodes marked for deletion.
  std::unordered_set<torch::jit::Node *> to_delete;
  for (torch::jit::Node *node : graph->nodes()) {
    if (isMarkedForDeletion(node)) {
      to_delete.insert(node);
    }
  }

  // Remove the dead nodes.
  searchAndPossiblyDestroy(to_delete);
}

} // namespace

void canonicalize(torch::jit::Graph *graph) {
  CanonicalizeImpl converter;
  converter.run(graph);
}

} // namespace poptorch
