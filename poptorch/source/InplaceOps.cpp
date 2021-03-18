// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>

#include <algorithm>
#include <utility>

#include "poptorch/InplaceOps.hpp"
#include "poptorch/InplaceOpsPyTorch.hpp_nolint"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/Utils.hpp"

namespace poptorch {

InplaceOpHandler::InplaceOpHandler(
    const std::shared_ptr<torch::jit::Graph> &graph, size_t num_parameters)
    : _graph(graph.get()) {
  _collapsed_inputs = collapsedGraphInputHierachy(graph.get());
  std::size_t num_inputs = graph->inputs().size() - num_parameters;

  // To begin with, none of the outputs are used for emulating inplacing.
  // Store the number of outputs (which may be nested) and the total number
  // of tensors output becase we will need these when handling the return values
  // from PopART to separate the original from additional outputs.
  _num_normal_outputs = graph->outputs().size();
  storeNumTensorOutputs();
  ERROR_ON(_collapsed_inputs.size() < num_inputs);

  // Now process each input and make changes if it is modified in place.
  for (size_t input = 0; input < num_inputs; input++) {
    processInput(input);
  }

  // There may still be inplace ops (which do not affect an input).
  // The original algorithm can safely handle these
  torch::jit::RemoveInplaceOps(graph);
}

void InplaceOpHandler::storeNumTensorOutputs() {
  _num_normal_tensor_outputs = 0;

  for (auto &output : _graph->outputs()) {
    if (output->node()->kind() == c10::prim::ListConstruct) {
      for (auto &input : output->node()->inputs()) {
        _num_normal_tensor_outputs += numTensorsForType(input->type());
      }
    } else {
      _num_normal_tensor_outputs += numTensorsForType(output->type());
    }
  }
}

void InplaceOpHandler::processInput(size_t input_num) {
  torch::jit::Value *current_alias = _collapsed_inputs[input_num];
  std::vector<torch::jit::Node *> to_delete;

  // Pass through the nodes in topological order rather than jumping to inputs
  for (auto node : _graph->nodes()) {
    // Skip if not in-place
    if (!torch::jit::isInplaceOp(node)) {
      continue;
    }

    auto inputs = node->inputs();
    ERROR_ON(inputs.empty());
    if (inputs[0] != current_alias) {
      continue;
    }

    auto new_kind = torch::jit::inPlaceToOutOfPlace.at(node->kind());

    torch::jit::WithInsertPoint insert_point(node);
    auto new_node = _graph->create(new_kind);
    _graph->insertNode(new_node);

    for (auto input : inputs) {
      new_node->addInput(input);
    }

    torch::jit::addAdditionalInputsIfRequired(_graph, node, new_node);

    current_alias = new_node->output();
    current_alias->setType(node->output()->type());

    node->output()->replaceAllUsesWith(current_alias);
    node->input(0)->replaceAllUsesAfterNodeWith(node, current_alias);

    to_delete.push_back(node);
  }

  for (auto node : to_delete) {
    node->destroy();
  }

  size_t output_mapping = InplaceOpHandler::no_mapping;

  if (current_alias != _collapsed_inputs[input_num]) {
    bool already_output = false;
    for (size_t output = 0; output < _graph->outputs().size(); output++) {
      if (_graph->outputs()[output] == current_alias) {
        already_output = true;
        output_mapping = output;
      }
    }

    if (!already_output) {
      output_mapping = _graph->registerOutput(current_alias);
    }
  }
  _input_output_mapping.push_back(output_mapping);
}

} // namespace poptorch
