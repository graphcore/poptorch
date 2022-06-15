// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

/*
 * CPU operations come in from the host in the form:

    Tensor[] %outputs = poptorch::call_cpu_op(%inputs)

    ... # Any traced user operations (to keep the trace consistent/happy)

    = poptorch::end_cpu_op(%8)

  * We need to do two things. Firstly we need to cull all the ops inbetween
  * call_cpu_op and end_cpu_op. Secondly we need to map the users of
  * poptorch::end_cpu_op to the outputs of poptorch::call_cpu_op.
  *
  * To do this we simply traverse through the nodes and record when we enter and
  * exit CPU op scope i.e between poptorch::call_cpu_op a poptorch::end_cpu_op.
*/

// extern c10::Symbol call_cpu_op;

namespace poptorch {

void cpuOffloadingCleanup(torch::jit::Graph *graph) {
  std::unordered_set<torch::jit::Node *> to_delete;

  // For diagnostics.
  std::size_t cpu_ops_found = 0;

  // The CPU op we are currently working on.
  torch::jit::Node *cpu_op_in_scope = nullptr;

  // For all nodes in the IR.
  for (torch::jit::Node *node : graph->nodes()) {
    const torch::jit::Symbol kind = node->kind();

    // Start CPU op scope.
    if (kind == symbols::poptorch::call_cpu_op) {
      ERROR_ON_MSG(
          cpu_op_in_scope != nullptr,
          "Trying to enter CPU from another CPU op! CPU ops must not overlap.");
      cpu_ops_found++;
      cpu_op_in_scope = node;
    } else if (kind == symbols::poptorch::canonicalised_cpu_call) {
      ERROR_ON_MSG(
          cpu_op_in_scope != nullptr,
          "Trying to enter CPU from another CPU op! CPU ops must not overlap.");
      cpu_ops_found++;
      cpu_op_in_scope = node;
    } else if (kind == symbols::poptorch::end_cpu_op) {
      to_delete.insert(node);

      // The form should be that the `end_cpu_op` feeds into a `ListUnpack` node
      // which converts the single output of the `end_cpu_op` (representing a
      // tuple/list) into multiple outputs. We transform it to eliminate that
      // unpack.
      torch::jit::Value *output = node->output();

      std::vector<torch::jit::Use> uses = output->uses();

      ERROR_ON_MSG(
          uses.empty(),
          "[Internal compiler error] CPU operation output has no uses.");
      ERROR_ON_MSG(
          uses.size() > 1,
          "[Internal compiler error] CPU operation output has multiple uses.");

      // List unpack
      torch::jit::Node *unpack = uses[0].user;
      ERROR_ON_MSG(unpack->kind() != c10::prim::ListUnpack,
                   "[Internal compiler error] CPU operation output is not used "
                   "by a list unpack");

      unpack->removeAllInputs();

      ERROR_ON_MSG(cpu_op_in_scope == nullptr,
                   "[Internal compiler error] CPU operation is null");

      // Remove the output.
      // Add the outputs and remap them to point to what the unpack previously
      // was used in.
      for (torch::jit::Value *old_out : unpack->outputs()) {
        torch::jit::Value *new_out = cpu_op_in_scope->addOutput();

        new_out->copyMetadata(old_out);
        old_out->replaceAllUsesWith(new_out);
      }

      // Remove the unpack.
      to_delete.insert(unpack);

      // Leave CPU scope.
      cpu_op_in_scope = nullptr;

    } else if (cpu_op_in_scope != nullptr) {
      // Unfortunately the compiler can put some non-functional SSA unpack ops
      // in the CPU scope that do logically outlive it.
      if (node->kind() != c10::prim::ListUnpack) {
        // Enables us to clean up some nodes without invalidating the IR.
        node->removeAllInputs();

        // Record the op for removal.
        to_delete.insert(node);
      }
    }
  }

  logging::trace("Found {} cpu ops. Removed {} nodes", cpu_ops_found,
                 to_delete.size());

  // Remove the dead nodes.
  searchAndPossiblyDestroy(to_delete);
}

} // namespace poptorch
