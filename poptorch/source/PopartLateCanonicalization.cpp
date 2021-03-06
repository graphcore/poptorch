// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <functional>

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

using FunctionTy = std::function<void()>;

// broadcast the scalar option vector to match num_convs
template <typename T> void broadcast(std::vector<T> &option, size_t num_convs) {
  if (option.size() != 1 || num_convs == 1) {
    return;
  }

  option.insert(option.end(), num_convs - 1, option[0]);
}

class MultiConvHandler {
public:
  explicit MultiConvHandler(torch::jit::Graph *g) : _graph(g) {}

  bool inMultiConv() const { return _in_multi_conv; }

  void begin(torch::jit::Node *node) {
    ERROR_ON_MSG(inMultiConv(), "Nested poptorch.MultiConv is not supported.");
    _in_multi_conv = true;
    _marker = node;
    _to_delete.insert(node);
  }

  void part(torch::jit::Node *node) { _parts.push_back(node); }

  FunctionTy end(torch::jit::Node *node) {
    ERROR_ON_MSG(!inMultiConv() || _parts.empty(),
                 "Unexpected end_multi_conv, is the IR malformed?");
    _in_multi_conv = false;
    applyOptions(node);
    return [this, node]() { applyPartLinks(node); };
  }

  void cleanup() { searchAndPossiblyDestroy(_to_delete); }

private:
  void applyOptions(torch::jit::Node *end_node) {
    // Fold any supplied options as attributes of the end_node.
    // Mark all options for deletion when we cleanup the IR.
    // available_memory_proportions
    torch::jit::Node *available_mem_props = end_node->input(0)->node();
    _to_delete.insert(available_mem_props);
    if (!isNone(available_mem_props)) {
      std::vector<double> vals =
          constantListToVec<double>(available_mem_props, constantToFloat);
      broadcast(vals, _parts.size());
      end_node->fs_(c10::Symbol::attr("available_memory_proportions"), vals);
    }

    // partials_types
    torch::jit::Node *partials_types = end_node->input(1)->node();
    _to_delete.insert(partials_types);
    if (!isNone(partials_types)) {
      std::vector<int64_t> vals = constantToLongVec(partials_types);
      broadcast(vals, _parts.size());
      end_node->is_(c10::Symbol::attr("partials_types"), vals);
    }

    // plan_type
    torch::jit::Node *plan_type = end_node->input(2)->node();
    _to_delete.insert(plan_type);
    if (!isNone(plan_type)) {
      end_node->i_(c10::Symbol::attr("plan_type"), constantToLong(plan_type));
    }

    // per_conv_reserved_tiles
    torch::jit::Node *reserved_tiles = end_node->input(3)->node();
    _to_delete.insert(reserved_tiles);
    if (!isNone(reserved_tiles)) {
      end_node->i_(c10::Symbol::attr("per_conv_reserved_tiles"),
                   constantToLong(reserved_tiles));
    }

    // cycle_back_off
    torch::jit::Node *back_off = end_node->input(4)->node();
    _to_delete.insert(back_off);
    if (!isNone(back_off)) {
      end_node->f_(c10::Symbol::attr("cycle_back_off"),
                   constantToFloat(back_off));
    }

    // Clear all the options from the end node inputs as they are now
    // incorporated as node attributes
    end_node->removeAllInputs();
  }

  void applyPartLinks(torch::jit::Node *end_node) {
    // Swaps out conv nodes with multi_conv_part which are then linked to the
    // end_node.  Each conv output flows through the end_multi_conv instruction.
    uint64_t num_outputs = 0;

    for (torch::jit::Node *node : _parts) {
      // Create the multi_conv_part node and update the marker ensures that the
      // all components of the multiconv are sequential in the IR.
      torch::jit::Node *conv_part = createMultiConvPart(_graph, node);
      conv_part->moveAfter(_marker);
      _marker = conv_part;
      _to_delete.insert(node);

      // Attach the multi_conv_part to the end_multi_conv instruction.
      end_node->addInput(conv_part->output());
      torch::jit::Value *output_i = end_node->addOutput();
      output_i->setType(conv_part->output()->type());
      replaceOutputUse(node->output(), end_node->output(num_outputs));
      num_outputs++;
    }

    end_node->moveAfter(_marker);
    _parts.clear();
    _marker = nullptr;
  }

  torch::jit::Graph *_graph;
  std::unordered_set<torch::jit::Node *> _to_delete;
  std::vector<torch::jit::Node *> _parts;
  torch::jit::Node *_marker = nullptr;
  bool _in_multi_conv = false;
};

void canonicalizeLate(torch::jit::Graph *graph) {
  /*
   * Perform the operation by looking for nodes we know need to be patched and
   * add the patching code to the callback which then all get called at once.
   * (To perserve the iterators.)
   */
  std::vector<FunctionTy> callbacks;
  MultiConvHandler multi_conv_handler(graph);

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
    } else if (kind == symbols::poptorch::begin_multi_conv) {
      multi_conv_handler.begin(node);
    } else if (multi_conv_handler.inMultiConv() &&
               kind == symbols::popart::conv) {
      multi_conv_handler.part(node);
    } else if (kind == symbols::poptorch::end_multi_conv) {
      callbacks.emplace_back(multi_conv_handler.end(node));
    }
  }

  // Execute the patchups.
  for (auto &callback : callbacks) {
    callback();
  }

  multi_conv_handler.cleanup();
}

} // namespace poptorch
