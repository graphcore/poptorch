// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

/*
 * No lint due to the linter expecting PopartLateCanonicalization.hpp which
 * rightly doesn't exist.
 */
#include <functional> // NOLINT

#include <poptorch/OpBuilder.hpp>              // NOLINT
#include <poptorch/PopartCanonicalization.hpp> // NOLINT
#include <torch/csrc/jit/ir/ir.h>              // NOLINT

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
    const std::string kindAsStr = kind.toDisplayString();

    /*
     * Pytorch group normalization allows for the user to either select the C
     * dimension or the W dimension so this needs to be rearanged. (Assuming
     * dimensions are [B, C, H, W] pytorch also allows <4 unlike poplibs).
     */
    if (kindAsStr == "popart::groupnormalization") {
      callbacks.push_back([node, &graph]() {
        torch::jit::Value *val = node->inputs()[0];

        c10::TensorTypePtr convertedToTensor =
            node->output()->type()->cast<c10::TensorType>();
        c10::VaryingShape dims = convertedToTensor->sizes();

        // Convert that IR type into a C++ vector of ints and insert an extra
        // dimension.
        std::vector<std::int64_t> newShape;
        std::vector<std::int64_t> originalShape;

        for (auto optionalInt : *dims.sizes()) {
          newShape.push_back(*optionalInt);
          originalShape.push_back(*optionalInt);
        }

        // If the dimensions are 3 add a singleton.
        if (newShape.size() == 3) {
          newShape.insert(newShape.begin(), 1);
        }

        // Reshape into 4D tensor.
        torch::jit::Node *reshaped = CreateReshape(graph, val, newShape);
        reshaped->insertBefore(node);

        // Transpose H to C.
        torch::jit::Node *transposed =
            Create_transpose(graph, {reshaped->output()}, {0, 3, 2, 1});
        transposed->insertAfter(reshaped);
        node->replaceInput(0, transposed->output());

        // Transpose C back to H in the output.
        torch::jit::Node *postTranspose =
            Create_transpose(graph, {node->output()}, {0, 3, 2, 1});
        postTranspose->insertAfter(node);

        // Reshape back to the original shape.
        torch::jit::Node *reshapeBackToOriginal =
            CreateReshape(graph, postTranspose->output(), originalShape);

        reshapeBackToOriginal->insertAfter(postTranspose);
        node->replaceAllUsesWith(reshapeBackToOriginal);

        // Replace all uses doesn't check that the use isn't in the instruction
        // doing the replacing! So we revert that manually.
        reshapeBackToOriginal->replaceInput(0, node->output());

        // We also want to keep the reference in the transpose op.
        postTranspose->replaceInput(0, node->output());
      });
      //}
    } else if (kindAsStr == "popart::slice") {
      /*
         Popart slice leaves in singleton dimensions whereas pytorch does not.
         So we must reshape the output to retain the pytorch form.
      */
      callbacks.push_back([node, &graph]() {
        c10::TensorTypePtr asTensor =
            node->output()->type()->cast<c10::TensorType>();
        c10::VaryingShape dims = asTensor->sizes();
        std::size_t dimensions = *dims.size();

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
    } else if (kindAsStr == "popart::nllloss") {
      callbacks.push_back([node, &graph]() {
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
