// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <utility>

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/InplaceOps.hpp"
#include "poptorch/InplaceOpsPyTorch.hpp_nolint"
#include "poptorch/OpBuilder.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"

#include "poptorch/Utils.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {

namespace {
namespace aten = c10::aten;
// Ops which only have an in-place version
const std::unordered_set<torch::jit::NodeKind> &onlyInplaceOps() {
  // static to make sure values are initialised
  static const std::unordered_set<torch::jit::NodeKind> only_inplace = {
      aten::copy_, aten::normal_, aten::uniform_, aten::random_,
      aten::exponential_};
  return only_inplace;
}

// Known view operations
const std::unordered_set<torch::jit::NodeKind> &viewOps() {
  // static to make sure values are initialised
  static const std::unordered_set<torch::jit::NodeKind> view_ops = {
      aten::chunk,    aten::detach,     aten::narrow,   aten::permute,
      aten::reshape,  aten::select,     aten::slice,    aten::split,
      aten::squeeze,  aten::transpose,  aten::unbind,   aten::unsqueeze,
      aten::view,     aten::as_strided, aten::diagonal, aten::movedim,
      aten::swapaxes, aten::swapdims,   aten::view_as,  aten::_unsafe_view};
  return view_ops;
}

size_t countNumTensorOutputs(torch::jit::Graph &graph) {
  size_t num_tensors = 0;

  for (const auto &output : graph.outputs()) {
    if (output->node()->kind() == c10::prim::ListConstruct) {
      for (const auto &input : output->node()->inputs()) {
        num_tensors += numTensorsForType(input->type());
      }
    } else {
      num_tensors += numTensorsForType(output->type());
    }
  }
  return num_tensors;
}

// When replacing `node` with `new_node`, if `new_node` doesn't have enough
// inputs pad them out with None-nodes.
// NOTE: Body mostly taken from torch (see torch::jit::RemoveInplaceOps), with
// the addition of metadata.
void addAdditionalInputsIfRequired(torch::jit::Graph *graph,
                                   const torch::jit::Node *node,
                                   torch::jit::Node *new_node) {
  int additional_input_count = 0;
  if (torch::jit::expectedInputCount.find(node->kind()) !=
      torch::jit::expectedInputCount.end()) {
    additional_input_count = torch::jit::expectedInputCount.at(node->kind()) -
                             static_cast<int>(new_node->inputs().size());
  }

  const WithNodeMetadata meta(new_node);
  for (int i = 0; i < additional_input_count; ++i) {
    auto *none_node = graph->createNone();
    // NOLINTNEXTLINE readability-suspicious-call-argument
    insertNodeBeforeNode(none_node, new_node);
    new_node->addInput(none_node->output());
  }
}

torch::jit::Node *outplaceOp(torch::jit::Graph &graph, torch::jit::Node *node) {
  torch::jit::NodeKind const new_kind = outplaceKind(node->kind());

  torch::jit::WithInsertPoint const insert_point(node);
  const WithNodeMetadata meta(node);
  auto *new_node = createAndInsertNode(&graph, new_kind, node->inputs());

  addAdditionalInputsIfRequired(&graph, node, new_node);

  new_node->output()->setType(node->output()->type());
  node->output()->replaceAllUsesWith(new_node->output());

  return new_node;
}

void removeRemainingInplaceOps(torch::jit::Graph &graph) {
  std::vector<torch::jit::Node *> to_delete;
  for (auto *node : graph.nodes()) {
    // Skip if not in-place
    if (!torch::jit::isInplaceOp(node)) {
      continue;
    }

    // Keep it in place if there is only an inplace version
    if (onlyInplaceOps().count(node->kind()) != 0) {
      continue;
    }

    outplaceOp(graph, node);
    to_delete.push_back(node);
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

} // namespace

torch::jit::NodeKind outplaceKind(torch::jit::NodeKind kind) {
  if (onlyInplaceOps().count(kind) != 0) {
    return kind;
  }

  std::string kind_str = kind.toQualString();

  torch::jit::NodeKind new_kind = kind;
  if (torch::jit::inPlaceToOutOfPlace.count(kind) != 0) {
    new_kind = torch::jit::inPlaceToOutOfPlace.at(kind);
  } else if (kind_str.back() == '_') {
    // Remove trailing '_' from the kind string
    kind_str.pop_back();
    new_kind = c10::Symbol::fromQualString(kind_str);
  }

  return new_kind;
}

void InplaceInputsTracker::addTensor(torch::jit::Value *input) {
  logging::trace("Tracking tensor %{}", input->debugName());

  const bool success = _aliases.insert({input, input}).second;
  ERROR_ON_MSG(!success, "Value already tracked");
}

torch::jit::Value *
InplaceInputsTracker::eraseCurrentAlias(torch::jit::Value *alias) {
  ERROR_ON(alias == nullptr);
  // Walk through the view ops until we find an input tensor.
  while (viewOps().count(alias->node()->kind()) != 0) {
    alias = alias->node()->input(0);
  }

  auto it = _aliases.find(alias);
  if (it != _aliases.end()) {
    auto *real_input = it->second;
    logging::trace("Deleted alias %{} for input %{}", it->first->debugName(),
                   it->second->debugName());
    // Remove current alias.
    _aliases.erase(it);
    return real_input;
  }
  return nullptr;
}

void InplaceInputsTracker::registerAlias(torch::jit::Value *aliased_input,
                                         torch::jit::Value *alias) {
  logging::trace("Registering alias %{} for input %{}", alias->debugName(),
                 aliased_input->debugName());
  ERROR_ON(!_aliases.insert({alias, aliased_input}).second);
}

InplaceGraphInfo
InplaceInputsTracker::finalizeGraph(torch::jit::Graph &graph,
                                    size_t num_anchors,
                                    bool replicas_needing_broadcast) {
  // For every alias (ie. target of an inplace op), look back and see if it's
  // applied through a bunch of views back to an input. if it is, mark it to be
  // handled later, at canonicalisation.
  for (const auto &[alias, aliased_input] : _aliases) {
    if (alias == aliased_input) {
      continue;
    }

    auto *inplace_op = alias->node();

    // Aliases are already traced back through views to graph inputs when
    // they're updated via `eraseCurrentAlias`, so can just check that the
    // ultimate input (`aliased_input`) is different to the inplace op's
    // immediate input.
    const bool was_inplace_on_view =
        !inplace_op->inputs().empty() && aliased_input != inplace_op->input(0);
    inplace_op->i_(c10::Symbol::attr("was_inplace_on_view"),
                   was_inplace_on_view ? 1 : 0);
  }

  // _aliases[alias] = graph_input -> we want the other way around.
  std::map<torch::jit::Value *, torch::jit::Value *> input_aliases;
  for (auto &p : _aliases) {
    ERROR_ON_MSG(!input_aliases.insert({p.second, p.first}).second,
                 "More than one alias for graph input %"
                     << p.second->debugName());
  }
  const size_t num_normal_tensor_outputs = countNumTensorOutputs(graph);
  InplaceGraphInfo out;
  out.num_normal_outputs = graph.outputs().size() + num_anchors;
  out.num_tensor_outputs = num_normal_tensor_outputs + num_anchors;

  const std::vector<torch::jit::Value *> collapsed_inputs =
      collapsedGraphInputHierachy(&graph);
  out.input_output_mapping.reserve(collapsed_inputs.size());
  for (const auto &graph_input : collapsed_inputs) {
    auto it = input_aliases.find(graph_input);
    ERROR_ON(it == input_aliases.end());
    size_t output_mapping = InplaceGraphInfo::no_mapping;
    if (it->first == it->second) {
      // no alias found
    } else {
      auto *alias = it->second;
      if (isParameter(graph_input)) {
        logging::trace("Alias for parameter %{} -> %{}", it->first->debugName(),
                       alias->debugName());
        // This is not supported with replicas needing broadcast
        ERROR_ON_MSG(
            replicas_needing_broadcast,
            "PopTorch does not support broadcasting buffers. If your "
            "model is able to tolerate buffers becoming out of sync "
            "between replicas, you can disable buffer broadcasting using "
            "poptorch.Options.broadcastBuffers(False).");

        const WithNodeMetadata meta(alias->node());
        auto *new_node =
            createAndInsertNode(&graph, symbols::poptorch::update_param_inplace,
                                {graph_input, alias});
        new_node->moveAfter(alias->node());
        new_node->output()->setType(alias->type());
      } else {
        logging::trace("Alias for input %{} -> %{}", it->first->debugName(),
                       alias->debugName());
        // Check if the alias is already being returned.
        for (size_t output = 0; output < graph.outputs().size(); output++) {
          if (graph.outputs()[output] == alias) {
            output_mapping = output;
          }
        }
        // If not, add a new output.
        if (output_mapping == InplaceGraphInfo::no_mapping) {
          output_mapping = graph.registerOutput(alias);

          // Ensure the overlap flag is set to no overlap (any models wanting
          // the additional efficiency of overalpped host IO should not use
          // inplace ops.)
          auto overlap_symbol =
              getOverlapSymbol("output", graph.outputs().size() - 1);
          graph.return_node()->s_(overlap_symbol, "no_overlap");
        }
      }
    }

    // The input/output mapping is only for 'true' inputs -- not parameters &
    // buffers (see its usage in PoplarExecutable::run).
    if (!isParameter(graph_input)) {
      out.input_output_mapping.push_back(output_mapping);
    }
  }

  // Outplace all the ops we can; the _aliases map no longer needs to be kept
  // up-to-date.
  removeRemainingInplaceOps(graph);

  return out;
}

void fixForLoopInputs(torch::jit::Graph &graph) {
  torch::jit::Value *correct_loop_input = nullptr;
  for (auto *node : graph.nodes()) {
    if (node->kind() == symbols::poptorch::start_for_loop) {
      ERROR_ON_MSG(correct_loop_input,
                   "[Internal] new poptorch::start_for_loop "
                   "encountered before previous poptorch::end_for_loop");
      correct_loop_input = node->input();
    } else if (node->kind() == symbols::poptorch::end_for_loop) {
      ERROR_ON_MSG(!correct_loop_input,
                   "[Internal] poptorch::end_for_loop "
                   "encountered before poptorch::start_for_loop");
      node->replaceInput(1, correct_loop_input);
      correct_loop_input = nullptr;
    }
  }
}

void verifyIfElseBlocksOrder(const torch::jit::Graph &graph) {
  // Verifies order of if...else blocks and generates friendly user error
  // messages if the order is incorrect.
  size_t if_cnt = 0;
  size_t else_cnt = 0;
  size_t end_cnt = 0;
  for (const auto *node : graph.nodes()) {
    if (node->kind() == symbols::poptorch::start_if_block) {
      if_cnt++;
    } else if (node->kind() == symbols::poptorch::start_else_block) {
      ERROR_ON_MSG(if_cnt <= else_cnt,
                   "[Internal] new poptorch::start_else_block "
                   "encountered before previous poptorch::start_if_block");
      else_cnt++;
    } else if (node->kind() == symbols::poptorch::end_if_block) {
      ERROR_ON_MSG(if_cnt < else_cnt || else_cnt <= end_cnt,
                   "[Internal] poptorch::end_if_block "
                   "encountered before poptorch::start_if_block and "
                   "poptorch::start_else_block");
      end_cnt++;
    }
  }
  ERROR_ON_MSG(!(if_cnt == else_cnt && else_cnt == end_cnt),
               "[Internal] no enclosing poptorch::end_if_block encountered");
}

} // namespace poptorch
