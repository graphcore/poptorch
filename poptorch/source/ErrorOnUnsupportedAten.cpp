// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

void warnOnUnsupportedAten(torch::jit::Graph *graph) {
  // Check that all of the "aten::" ops have been eliminated.
  std::unordered_set<torch::jit::Symbol> unsupported_ops;

  for (torch::jit::Node *node : graph->nodes()) {
    if (node->kind().is_aten()) {
      unsupported_ops.insert(node->kind());
    }
  }

  // Terminate compilation via error.
  if (!unsupported_ops.empty()) {
    std::stringstream ss;
    std::string sep;
    for (const auto &op : unsupported_ops) {
      ss << sep << op.toQualString();
      sep = ", ";
    }

    ERROR("Unsupported ops found in compiled model: ["
          << ss.str()
          << "]. Not all operations are supported yet by Graphcore's PyTorch "
             "compiler. If you believe any of these should be, please report "
             "this message to support@graphcore.ai.");
  }
}

} // namespace poptorch
