// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

void warnOnUnsupportedAten(torch::jit::Graph *graph) {
  // Check that all of the "aten::" ops have been eliminated.
  bool has_unsupported_op = false;

  std::unordered_set<std::string> previously_reported_node;
  for (torch::jit::Node *node : graph->nodes()) {
    const torch::jit::Symbol kind = node->kind();

    if (kind.is_aten()) {
      const std::string domain = kind.toQualString();

      if (previously_reported_node.count(domain) == 0) {
        logging::warn(
            "Unsupported operation found in compiled module: {}. Not all "
            "operations are supported yet by Graphcore's pytorch compiler. "
            "If you believe this one should be, please report this message to "
            "support@graphcore.ai.",
            domain);

        previously_reported_node.insert(domain);
        has_unsupported_op = true;
      }
    }
  }

  // Terminate compilation via error.
  if (has_unsupported_op) {
    ERROR("Unsupported ops found in compiled model (see warning log).");
  }
}

} // namespace poptorch
