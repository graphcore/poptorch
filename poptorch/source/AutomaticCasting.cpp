// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>

#include "torch/csrc/jit/ir/ir.h"

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/ImplicitCasting.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {

class AutocastPolicy {
public:
  AutocastPolicy() : is_enabled(false) {}

  bool decision(const torch::jit::Node *node, at::ScalarType *type) const;

  bool enabled() const { return is_enabled; }

  void enabled(bool value) {
    logging::debug("poptorch.Options set autocastEnabled to {}", value);
    is_enabled = value;
  }

  static void initSet(std::set<std::string> *set,
                      const std::vector<std::string> &ops) {
    set->clear();
    for (const auto &op : ops) {
      set->insert(op);
    }
  }

  void initHalf(const std::vector<std::string> &ops) {
    initSet(&fp16, ops);

    std::string set_image;
    renderSet(fp16, &set_image);
    logging::debug("Automatic casting policy fp16 set to: {}", set_image);
  }

  void initFloat(const std::vector<std::string> &ops) {
    initSet(&fp32, ops);

    std::string set_image;
    renderSet(fp32, &set_image);
    logging::debug("Automatic casting policy fp32 set to: {}", set_image);
  }

  void initPromote(const std::vector<std::string> &ops) {
    initSet(&promote, ops);

    std::string set_image;
    renderSet(promote, &set_image);
    logging::debug("Automatic casting policy promote set to: {}", set_image);
  }

  void initDemote(const std::vector<std::string> &ops) {
    initSet(&demote, ops);

    std::string set_image;
    renderSet(demote, &set_image);
    logging::debug("Automatic casting policy demote set to: {}", set_image);
  }

protected:
  bool is_enabled;
  std::set<std::string> fp16, fp32, promote, demote;

  static void renderSet(const std::set<std::string> &set, std::string *str) {
    if (set.empty()) {
      *str = "[ (empty) ]";
      return;
    }

    *str = "[";

    for (const auto &s : set) {
      *str += " " + s;
    }

    *str += " ]";
  }
};

bool valueHasScalarType(const torch::jit::Value *value,
                        at::ScalarType *scalar_type) {
  auto tensor_type = value->type()->cast<c10::TensorType>();
  if (!tensor_type || !tensor_type->scalarType()) {
    return false;
  }

  *scalar_type = *tensor_type->scalarType();
  return true;
}

// Given a JIT node, decide whether its type should be changed.
// If so, return true and set type to the desired scalar type.
// Otherwise return false;
bool AutocastPolicy::decision(const torch::jit::Node *node,
                              at::ScalarType *type) const {
  bool mixed_precision = false;
  bool has_float = true;
  bool has_half = false;
  std::string kind = node->kind().toUnqualString();
  at::ScalarType scalar_type;

  // determine whether node has mixed precision inputs or outputs
  for (auto input : node->inputs()) {
    if (!valueHasScalarType(input, &scalar_type)) {
      continue;
    }

    has_float = has_float || scalar_type == at::ScalarType::Float;
    has_half = has_float || scalar_type == at::ScalarType::Half;
  }

  for (auto output : node->outputs()) {
    if (!valueHasScalarType(output, &scalar_type)) {
      continue;
    }

    has_float = has_float || getNodeScalarType(output) == at::ScalarType::Float;
    has_half = has_float || getNodeScalarType(output) == at::ScalarType::Half;
  }

  mixed_precision = has_float && has_half;

  // for mixed precision nodes, try to apply promote or demote rule first
  if (mixed_precision) {
    auto it = promote.find(kind);
    if (it != promote.end()) {
      *type = at::ScalarType::Float;
      return true;
    }

    it = demote.find(kind);
    if (it != demote.end()) {
      *type = at::ScalarType::Half;
      return true;
    }
  }

  // try to apply fp16 or fp32 rules
  auto it = fp16.find(kind);
  if (it != fp16.end()) {
    *type = at::ScalarType::Half;
    return true;
  }

  it = fp32.find(kind);
  if (it != fp32.end()) {
    *type = at::ScalarType::Float;
    return true;
  }

  // if we got this far with a mixed-precision node, apply the implicit
  // casting policy now
  if (mixed_precision) {
    switch (halfFloatCastingBehavior()) {
    case HalfFloatCasting::FloatDowncastToHalf:
      *type = at::ScalarType::Half;
      break;
    case HalfFloatCasting::HalfUpcastToFloat:
      *type = at::ScalarType::Float;
      break;
    default:
      throw std::runtime_error(
          "Unsupported HalfFloatCastingBehavior in AutocastPolicy::decision");
    }
    return true;
  }

  return false;
}

// The policy object
AutocastPolicy policy;
// map original values in their auto-cast counterparts
std::map<torch::jit::Value *, torch::jit::Value *> cast_value;

torch::jit::Value *castValue(torch::jit::Graph *graph, torch::jit::Value *value,
                             at::ScalarType type) {
  // no cast needed
  if (getNodeScalarType(value) == type) {
    return value;
  }

  // reuse any previously created nodes
  auto it = cast_value.find(value);
  if (it != cast_value.end()) {
    return it->second;
  }

  // create a poptorch::autocast node
  auto current_type = value->type()->cast<c10::TensorType>();
  auto new_node = graph->create(symbols::poptorch::autocast);
  new_node->addInput(value);
  new_node->insertAfter(value->node());
  new_node->output(0)->setType(current_type->withScalarType(type));
  cast_value.insert(std::make_pair(value, new_node->output(0)));

  return new_node->output(0);
}

bool valueNeedsCast(const torch::jit::Value *value, at::ScalarType type) {
  at::ScalarType current_type;

  if (!valueHasScalarType(value, &current_type)) {
    return false;
  }

  if (current_type == type) {
    return false;
  }

  return current_type == at::ScalarType::Float ||
         current_type == at::ScalarType::Half;
}

// apply autocast policy for a node
void autocastNode(torch::jit::Graph *graph, torch::jit::Node *node) {
  at::ScalarType new_type;

  // obtain casting decision from policy object
  if (!policy.decision(node, &new_type)) {
    return;
  }

  // cast half / float inputs
  for (auto input : node->inputs()) {
    if (!valueNeedsCast(input, new_type)) {
      continue;
    }

    auto new_input = castValue(graph, input, new_type);
    node->replaceInputWith(input, new_input);
  }

  // change type of half / float outpouts
  for (auto output : node->outputs()) {
    if (!valueNeedsCast(output, new_type)) {
      continue;
    }

    auto tensor_type = output->type()->cast<c10::TensorType>();
    output->setType(tensor_type->withScalarType(new_type));
  }
}

} // namespace

void setAutocastEnabled(bool value) { policy.enabled(value); }
void setAutocastHalf(std::vector<std::string> &&ops) { policy.initHalf(ops); }
void setAutocastFloat(std::vector<std::string> &&ops) { policy.initFloat(ops); }
void setAutocastPromote(std::vector<std::string> &&ops) {
  policy.initPromote(ops);
}
void setAutocastDemote(std::vector<std::string> &&ops) {
  policy.initDemote(ops);
}

// Automatic casting control node types:
// 1. begin_autocast - marks the beginning of an autocast region
// 2. supress_autocast - begins a region where autocast is disabled
// 3. restore_autocast - ends an autocasting region
void automaticCasting(torch::jit::Graph *graph) {
  // stack corresponding to nested regions where autocasting is
  // disabled or enabled
  std::stack<int> in_autocast_region;
  // nodes to be removed at the end of the pass
  std::set<torch::jit::Node *> to_remove;
  // set of nodes in auto-casting regions
  std::set<torch::jit::Node *> autocast_nodes;

  cast_value.clear();
  to_remove.clear();
  in_autocast_region.push(0);

  // First pass: process nodes in auto-casting regions
  for (auto node : graph->nodes()) {
    // begin an autocasting region - increments the top of stack
    if (node->kind() == symbols::poptorch::begin_autocast) {
      in_autocast_region.top()++;
      to_remove.insert(node);
      continue;
    }

    // begins are new region where autocast is disabled
    // push 0 to the stack
    if (node->kind() == symbols::poptorch::suppress_autocast) {
      in_autocast_region.push(0);
      to_remove.insert(node);
      continue;
    }

    // concludes an autocasting region
    if (node->kind() == symbols::poptorch::restore_autocast) {
      if (in_autocast_region.top() == 0) {
        // we have cancelled out all begin_autocast nodes
        in_autocast_region.pop();
      } else {
        // end effect of the latest begin_autocast node
        in_autocast_region.top()--;
      }
      to_remove.insert(node);
      continue;
    }

    // autocast is enabled in this region iff top of stack is nonzero
    if (policy.enabled() && in_autocast_region.top() > 0) {
      autocastNode(graph, node);
    }
  }

  // Finally: remove autocast region markers
  for (auto node : to_remove) {
    node->destroy();
  }
}

} // namespace poptorch
