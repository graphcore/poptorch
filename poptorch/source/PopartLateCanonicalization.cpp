// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <functional>

#include <poptorch/OpBuilder.hpp>
#include <poptorch/PopartCanonicalization.hpp>

#include "PoptorchSymbols.hpp"

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
    } else if (kind == symbols::popart::nllloss) {
      callbacks.emplace_back([node]() {
        /*
         * NLLloss in popart performs the log operation whereas pytorch doesn't.
         */

        torch::jit::Node *log = node->inputs()[0]->node();
        const std::string log_as_str = log->kind().toDisplayString();

        // Make sure it is an log.
        if (log_as_str == "popart::log") {
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
