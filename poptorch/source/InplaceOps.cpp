// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <utility>

#include "poptorch/InplaceOps.hpp"
#include "poptorch/InplaceOpsPyTorch.hpp_nolint"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/Utils.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {

namespace {
// Ops which only have an in-place version
const std::unordered_set<torch::jit::NodeKind> &onlyInplaceOps() {
  // static to make sure values are initialised
  static std::unordered_set<torch::jit::NodeKind> only_implace = {
      c10::aten::copy_, c10::aten::normal_, c10::aten::uniform_};
  return only_implace;
}
} // namespace

InplaceOpHandler::InplaceOpHandler(
    const std::shared_ptr<torch::jit::Graph> &graph, size_t num_parameters,
    size_t num_anchors, bool replicas)
    : _graph(graph.get()), _num_anchors(num_anchors), _replicas(replicas) {
  _collapsed_inputs = collapsedGraphInputHierachy(graph.get());
  _num_tensor_inputs = _collapsed_inputs.size() - num_parameters;

  // To begin with, none of the outputs are used for emulating inplacing.
  // Store the number of outputs (which may be nested) and the total number
  // of tensors output becase we will need these when handling the return values
  // from PopART to separate the original from additional outputs.
  _num_normal_outputs = graph->outputs().size();
  storeNumTensorOutputs();

  // Now process each input and make changes if it is modified in place.
  for (size_t input = 0; input < _collapsed_inputs.size(); input++) {
    processInput(input);
  }

  // There may still be inplace ops (which do not affect an input).
  // These must also be removed.
  removeRemainingInplaceOps();
}

void InplaceOpHandler::storeNumTensorOutputs() {
  _num_normal_tensor_outputs = 0;

  for (const auto &output : _graph->outputs()) {
    if (output->node()->kind() == c10::prim::ListConstruct) {
      for (const auto &input : output->node()->inputs()) {
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

  // Pass through the nodes in topological order rather than jumping through
  // outputs
  for (auto *node : _graph->nodes()) {
    // Skip if not in-place
    if (!torch::jit::isInplaceOp(node)) {
      continue;
    }

    auto inputs = node->inputs();
    ERROR_ON(inputs.empty());
    if (inputs[0] != current_alias) {
      continue;
    }

    // Generally the type will match but a half may be cast to a float for
    // the sake of tracing. In the case of parameters, the input will already be
    // a half causing a mismatch here, which needs to be changed.
    auto current_tensor_type = current_alias->type()->cast<c10::TensorType>();
    auto output_tensor_type = node->output()->type()->cast<c10::TensorType>();
    if (current_tensor_type && output_tensor_type) {
      if (*current_tensor_type->scalarType() == at::ScalarType::Half &&
          *output_tensor_type->scalarType() == at::ScalarType::Float) {
        node->output()->setType(
            current_tensor_type->withScalarType(at::ScalarType::Half));
      }
      ERROR_ON(*current_tensor_type->scalarType() !=
               *output_tensor_type->scalarType());
    }

    // Keep it in place if there is only an inplace version
    if (onlyInplaceOps().count(node->kind()) != 0) {
      current_alias = node->output();
      continue;
    }

    auto *new_node = outplaceOp(node);
    to_delete.push_back(node);
    current_alias = new_node->output();
  }

  // Handle differently for normal inputs and parameters
  bool is_parameter = input_num >= _num_tensor_inputs;

  // Check if it is not modified in place at all
  bool is_modified_inplace = current_alias != _collapsed_inputs[input_num];
  if (!is_modified_inplace) {
    if (!is_parameter) {
      _input_output_mapping.push_back(InplaceOpHandler::no_mapping);
    }

    ERROR_ON(!to_delete.empty());
    return;
  }

  if (is_parameter) {
    // This is not supported with replicas
    ERROR_ON_MSG(_replicas, "Model modifies a buffer in place. This is not "
                            "supported when using replication i.e. "
                            "replicationFactor > 1 with poptorch.Options.\n"
                            "Last modification: "
                                << *current_alias->node());

    auto *new_node = _graph->create(symbols::poptorch::update_param_inplace, 1);
    new_node->addInput(_collapsed_inputs[input_num]);
    new_node->addInput(current_alias);
    new_node->insertAfter(current_alias->node());
    new_node->output()->setType(current_alias->type());

  } else {
    // Not a parameter : handle by adding it as an output which will be used
    // to update the input tensor
    size_t output_mapping = InplaceOpHandler::no_mapping;

    if (is_modified_inplace) {
      for (size_t output = 0; output < _graph->outputs().size(); output++) {
        if (_graph->outputs()[output] == current_alias) {
          output_mapping = output;
        }
      }
    }

    if (output_mapping == InplaceOpHandler::no_mapping) {
      output_mapping = _graph->registerOutput(current_alias);

      // Ensure the overlap flag is set to no overlap (any models wanting the
      // additional efficiency of overalpped host IO should not use inplace
      // ops.)
      auto overlap_symbol =
          getOverlapSymbol("_for_output", _graph->outputs().size() - 1);
      _graph->return_node()->s_(overlap_symbol, "no_overlap");
    }
    _input_output_mapping.push_back(output_mapping);
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

void InplaceOpHandler::removeRemainingInplaceOps() {
  std::vector<torch::jit::Node *> to_delete;
  for (auto *node : _graph->nodes()) {
    // Skip if not in-place
    if (!torch::jit::isInplaceOp(node)) {
      continue;
    }

    // Keep it in place if there is only an inplace version
    if (onlyInplaceOps().count(node->kind()) != 0) {
      continue;
    }

    outplaceOp(node);
    to_delete.push_back(node);
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

torch::jit::Node *InplaceOpHandler::outplaceOp(torch::jit::Node *node) {
  torch::jit::NodeKind new_kind;
  if (torch::jit::inPlaceToOutOfPlace.count(node->kind()) != 0) {
    new_kind = torch::jit::inPlaceToOutOfPlace.at(node->kind());
  } else {
    // Remove trailing '_' from the kind string
    std::string kind_str(node->kind().toQualString());
    std::string modified_kind_str = kind_str.substr(0, kind_str.length() - 1);
    new_kind = c10::Symbol::fromQualString(modified_kind_str);
  }

  torch::jit::WithInsertPoint insert_point(node);
  auto *new_node = _graph->create(new_kind);
  _graph->insertNode(new_node);

  for (auto *input : node->inputs()) {
    new_node->addInput(input);
  }

  torch::jit::addAdditionalInputsIfRequired(_graph, node, new_node);

  new_node->output()->setType(node->output()->type());
  node->output()->replaceAllUsesWith(new_node->output());
  node->input(0)->replaceAllUsesAfterNodeWith(node, node->output());

  return new_node;
}

} // namespace poptorch
