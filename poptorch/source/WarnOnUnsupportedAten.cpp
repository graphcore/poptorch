// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <poptorch/PopartCanonicalization.hpp>
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

namespace poptorch {

void WarnOnUnsupportedAten(torch::jit::Graph &graph) {
  // Check that all of the "aten::" ops have been eliminated.
  bool hasUnsupportedOp = false;
  for (torch::jit::Node *node : graph.nodes()) {
    const torch::jit::Symbol kind = node->kind();

    if (kind.is_aten()) {
      const std::string domain = kind.toQualString();

      logging::warn(
          "Unsupported operation found in compiled module: {}. Not all "
          "operations are supported yet by Graphcore's pytorch compiler. "
          "If you believe this one should be, please report this message to "
          "support@graphcore.ai.",
          domain);
      hasUnsupportedOp = true;
    }
  }

  // Terminate compilation via error.
  if (hasUnsupportedOp) {
    ERROR("Unsupported ops found in compiled model (see warning log).");
  }
}

} // namespace poptorch
