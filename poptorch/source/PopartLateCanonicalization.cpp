// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <functional>

#include "PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {

using FunctionTy = std::function<void()>;

void canonicalizeLate(torch::jit::Graph *graph) {
  /*
   * Perform the operation by looking for nodes we know need to be patched and
   * add the patching code to the callback which then all get called at once.
   * (To perserve the iterators.)
   */
  std::vector<FunctionTy> callbacks;

  // Look for the nodes.
  for (torch::jit::Node *node : graph->nodes()) {
    logging::LogContext ctx("canonicalizeLate Processing " +
                            nodeToString(node));
    const torch::jit::Symbol kind = node->kind();

    if (kind == symbols::popart::slice) {
      /*
         Popart slice leaves in singleton dimensions whereas pytorch does not.
         So we must reshape the output to retain the pytorch form.
      */
      callbacks.emplace_back([node, &graph]() {
        c10::TensorTypePtr as_tensor =
            node->output()->type()->cast<c10::TensorType>();

        c10::VaryingShape dims = as_tensor->sizes();

        if (!dims.size()) {
          return;
        }

        std::vector<std::int64_t> original_shape;

        for (auto optional_int : *dims.sizes()) {
          original_shape.push_back(*optional_int);
        }

        torch::jit::Node *reshaped =
            createReshape(graph, node->output(), original_shape);
        reshaped->moveAfter(node);

        node->replaceAllUsesWith(reshaped);

        // Replace all uses doesn't check that the use isn't in the instruction
        // doing the replacing! So we revert that manually.
        reshaped->replaceInput(0, node->output());

        // Take the type of the old value.
        reshaped->output()->setType(node->output()->type());
      });
    }
  }

  // Execute the patchups.
  for (auto &callback : callbacks) {
    callback();
  }
}

} // namespace poptorch
