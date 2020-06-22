// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

/*
 * No lint due to the linter expecting PopartLateCanonicalization.hpp which
 * rightly doesn't exist.
 */
#include <functional> // NOLINT

#include <poptorch/OpBuilder.hpp>              // NOLINT
#include <poptorch/PopartCanonicalization.hpp> // NOLINT
#include <torch/csrc/jit/ir/ir.h>              // NOLINT

#include "PoptorchSymbols.h"

namespace poptorch {

using FunctionTy = std::function<void()>;

void CanonicalizeLate(torch::jit::Graph &graph) {
  /*
   * Perform the operation by looking for nodes we know need to be patched and
   * add the patching code to the callback which then all get called at once.
   * (To perserve the iterators.)
   */
  std::vector<FunctionTy> callbacks;

  // Look for the nodes.
  for (torch::jit::Node *node : graph.nodes()) {
    const torch::jit::Symbol kind = node->kind();

    if (kind == Symbols::popart::slice) {
      /*
         Popart slice leaves in singleton dimensions whereas pytorch does not.
         So we must reshape the output to retain the pytorch form.
      */
      callbacks.push_back([node, &graph]() {
        c10::TensorTypePtr asTensor =
            node->output()->type()->cast<c10::TensorType>();

        c10::VaryingShape dims = asTensor->sizes();

        if (!dims.size())
          return;

        std::vector<std::int64_t> originalShape;

        for (auto optionalInt : *dims.sizes()) {
          originalShape.push_back(*optionalInt);
        }

        torch::jit::Node *reshaped =
            CreateReshape(graph, node->output(), originalShape);

        reshaped->insertAfter(node);
        node->replaceAllUsesWith(reshaped);

        // Replace all uses doesn't check that the use isn't in the instruction
        // doing the replacing! So we revert that manually.
        reshaped->replaceInput(0, node->output());

        // Take the type of the old value.
        reshaped->output()->setType(node->output()->type());
      });
    } else if (kind == Symbols::popart::nllloss) {
      callbacks.push_back([node]() {
        /*
         * NLLloss in popart performs the log operation whereas pytorch doesn't.
         */

        torch::jit::Node *log = node->inputs()[0]->node();
        const std::string logAsStr = log->kind().toDisplayString();

        // Make sure it is an log.
        if (logAsStr == "popart::log") {
          // Just use the softmax directly.
          node->replaceInputWith(log->output(), log->input());
        }
      });
    }
  }

  // Execute the patchups.
  for (auto &callback : callbacks) {
    callback();
  }
}

} // namespace poptorch
