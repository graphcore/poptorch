// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <map>
#include <queue>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {

void removeScatterAddIndexExpansion(torch::jit::Graph *graph) {
  const logging::LogContext ctx{"ScatterAddOptimization"};

  std::vector<torch::jit::Node *> to_delete;

  for (auto *node : graph->nodes()) {
    if (node->kind() != c10::aten::scatter_add &&
        node->kind() != c10::aten::scatter_add_ &&
        node->kind() != c10::aten::scatter_reduce &&
        node->kind() != c10::aten::scatter_reduce_) {
      continue;
    }

    auto *index = node->input(2);
    auto *index_producer = index->node();

    // Only remove index expansions.
    if (index_producer->kind() != c10::aten::expand &&
        index_producer->kind() != c10::aten::expand_as) {
      continue;
    }

    auto *src = node->input(3);
    auto *original_index = index_producer->input(0);
    const auto expanded_index_shape = shapeFromTensor(index);

    // Make sure removal is valid
    if (index->uses().size() > 1 ||
        shapeFromTensor(src) != expanded_index_shape) {
      continue;
    }

    logging::trace("Removing index expansion node: {}",
                   nodeToString(index_producer));
    node->replaceInputWith(index, original_index);
    to_delete.push_back(index_producer);
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

// Apply grouped version of scatter.
torch::jit::Node *applyScatter(torch::jit::Graph *graph,
                               torch::jit::Node *node_with_attributes,
                               const std::vector<torch::jit::Value *> &inputs,
                               int64_t num_scatters) {
  const auto axis_size =
      node_with_attributes->i(c10::Symbol::attr("axis_size"));
  const auto old_axis = node_with_attributes->i(c10::Symbol::attr("axis"));
  const auto reduction =
      node_with_attributes->i(c10::Symbol::attr("reduction"));
  return createGroupedscatterreduce(graph, inputs, axis_size, old_axis + 1,
                                    reduction, num_scatters);
}

torch::jit::Node *mergeSet(torch::jit::Graph *graph,
                           torch::jit::node_list &scatters, bool with_update) {
  std::unordered_map<torch::jit::Value *, torch::jit::Node *> srcs;
  std::unordered_set<torch::jit::Node *> to_destroy;

  auto *last_scatter = scatters[0];
  srcs.emplace(last_scatter->input(0), last_scatter);
  for (size_t i = 1; i < scatters.size(); i++) {
    auto *scatter = scatters[i];
    // If encountered, remove the repetition.
    if (auto input0_it = srcs.find(scatter->input(0));
        input0_it != srcs.end()) {
      // Check with second input.
      auto *scatter_with_same_input0 = input0_it->second;
      bool is_repetition =
          scatter->input(1) == scatter_with_same_input0->input(1);
      if (with_update) {
        is_repetition = is_repetition &&
                        scatter->input(2) == scatter_with_same_input0->input(2);
      }

      if (is_repetition) {
        replaceOutputUse(scatter->output(), scatter_with_same_input0->output());
        to_destroy.insert(scatter);
        scatters.erase(scatters.begin() + i);
        i--;
      }
    } else {
      srcs.emplace(scatter->input(0), scatter);
      // Find last scatter to execute. This should be insert point for grouped
      // scatter.
      if (last_scatter->isBefore(scatter)) {
        last_scatter = scatter;
      }
    }
  }

  const auto insert_outputs_users = [](torch::jit::Node *node_to_process,
                                       std::queue<torch::jit::Node *> &queue) {
    for (auto *output : node_to_process->outputs()) {
      for (const auto &use : output->uses()) {
        const auto &user = use.user;
        queue.push(user);
      }
    }
  };

  // Move nodes from scatters after grouped scatter
  auto *insert_point_node = last_scatter;
  for (size_t i = 0; i < scatters.size(); i++) {
    auto *scatter = scatters[i];
    if (scatter != last_scatter) {
      std::queue<torch::jit::Node *> nodes_to_move;
      insert_outputs_users(scatter, nodes_to_move);
      while (!nodes_to_move.empty()) {
        auto *node = nodes_to_move.front();
        nodes_to_move.pop();
        if (node->isBefore(insert_point_node)) {
          node->moveAfter(insert_point_node);
          insert_point_node = node;
          insert_outputs_users(node, nodes_to_move);
        }
      }
    }
  }

  std::vector<torch::jit::Value *> inputs0_scat(scatters.size());
  std::vector<torch::jit::Value *> inputs1_scat(scatters.size());
  std::vector<torch::jit::Value *> inputs2_scat;
  if (with_update) {
    inputs2_scat = std::vector<torch::jit::Value *>(scatters.size());
  }

  // Store inputs.

  for (size_t i = 0; i < scatters.size(); i++) {
    auto *scatter = scatters[i];
    inputs0_scat[i] = scatter->input(0);
    inputs1_scat[i] = scatter->input(1);
    if (with_update) {
      inputs2_scat[i] = scatter->input(2);
    }
  }

  const int64_t num_scatters = static_cast<int64_t>(scatters.size());

  const WithNodeMetadata meta{last_scatter};
  const torch::jit::WithInsertPoint insert_point(last_scatter);

  for (size_t i = 0; i < inputs0_scat.size(); i++) {
    inputs0_scat[i] = createUnsqueeze(graph, {inputs0_scat[i]}, {0})->output();
    inputs1_scat[i] = createUnsqueeze(graph, {inputs1_scat[i]}, {0})->output();
    if (with_update) {
      inputs2_scat[i] =
          createUnsqueeze(graph, {inputs2_scat[i]}, {0})->output();
    }
  }

  std::vector<torch::jit::Value *> args;
  args.reserve(3);

  args.push_back(createConcat(graph, inputs0_scat, 0)->output());
  args.push_back(createConcat(graph, inputs1_scat, 0)->output());

  if (with_update) {
    args.push_back(createConcat(graph, inputs2_scat, 0)->output());
  }

  auto *grouped_scatter = applyScatter(graph, scatters[0], args, num_scatters);

  for (int64_t i = 0; i < num_scatters; i++) {
    auto *slice =
        createSlice(graph, {grouped_scatter->output()}, {i + 1}, {i}, {0});
    auto *squeeze = createSqueeze(graph, {slice->output()}, {0});

    // Replace outputs with grouped version.
    auto *scatter_to_replace = scatters[i];
    for (auto *output : scatter_to_replace->outputs()) {
      replaceOutputUse(output, squeeze->output());
    }
    to_destroy.insert(scatter_to_replace);
  }
  // Destroy merged scatters.
  searchAndPossiblyDestroy(to_destroy);
  return grouped_scatter;
}

torch::jit::node_list dispatchScatters(torch::jit::Graph *graph,
                                       torch::jit::node_list &scatters,
                                       bool with_update) {
  using Shape = std::vector<std::int64_t /*dim*/>;
  using Group = std::tuple<std::int64_t /*axis*/, Shape /*index*/,
                           Shape /*src*/, Shape /*self*/>;

  std::map<Group, torch::jit::node_list> group_to_merge_candidates;

  for (torch::jit::Node *scatter_node : scatters) {

    const std::int64_t axis = scatter_node->i(c10::Symbol::attr("axis"));
    const Shape src_shape = shapeFromTensor(scatter_node->input(0));
    const Shape index_shape = shapeFromTensor(scatter_node->input(1));
    const Shape self_shape =
        with_update ? shapeFromTensor(scatter_node->input(2)) : Shape{};

    const Group key{axis, index_shape, src_shape, self_shape};
    group_to_merge_candidates[key].push_back(scatter_node);
  }

  torch::jit::node_list grouped_scatters;
  for (auto &&[_, merge_candidates] : group_to_merge_candidates) {

    if (merge_candidates.size() > 1) {
      grouped_scatters.push_back(
          mergeSet(graph, merge_candidates, with_update));
    } else {
      grouped_scatters.push_back(merge_candidates.front());
    }
  }

  return grouped_scatters;
}

/*
 * Algorithm:
 * 1. Move the BFS around the graph and add only those that all inputs are
 *    encountered until the entire queue is scatters.
 * 2. Merge the scatters.
 * 3. Add outputs to queue and remove scatters.
 * 4. If queue is not empty go to step 1.
 */
void fuseScatters(torch::jit::Graph *graph) {
  logging::LogContext const ctx{"fuseScatters"};

  // Queue contains fully reached nodes.
  std::queue<torch::jit::Node *> queue;
  // Add roots to queue.
  std::unordered_set<torch::jit::Node *> added;
  for (torch::jit::Node *node : graph->nodes()) {
    if (node->inputs().empty()) {
      if (added.find(node) == added.end()) {
        queue.push(node);
        added.insert(node);
      }
    }
  }
  for (torch::jit::Value *input : graph->inputs()) {
    auto *node = input->node();
    if (added.find(node) == added.end()) {
      queue.push(node);
      added.insert(node);
    }
  }

  // The unordered_map elements represent the number of times the node was
  // reached.
  std::unordered_map<torch::jit::Node *, std::size_t> node_num_visited_inputs;
  // The unordered_set elements mean that children have been added to the queue.
  // std::unordered_set<torch::jit::Node *> visited;

  using ScatterKind =
      std::tuple<std::int64_t /*reduction*/, at::ScalarType /*input_type*/,
                 bool /*with_update*/>;
  std::map<ScatterKind, torch::jit::node_list> scatters;

  std::size_t num_scatters_in_queue = 0;

  // Lambda to add the children of the vertex.
  const auto add_children_to_queue = [&](const torch::jit::Node *node) {
    for (const torch::jit::Value *output : node->outputs()) {
      for (const torch::jit::Use &use : output->uses()) {
        torch::jit::Node *user = use.user;
        const auto num_user_inputs = user->inputs().size();

        auto &num_user_visited_inputs = node_num_visited_inputs[user];
        ++num_user_visited_inputs;

        if (num_user_visited_inputs == num_user_inputs) {
          queue.push(user);
          if (user->kind() == symbols::popart::scatterreduce) {
            ++num_scatters_in_queue;
            const std::int64_t reduction =
                user->i(c10::Symbol::attr("reduction"));
            const at::ScalarType input_type = *user->input(0)
                                                   ->type()
                                                   ->expect<c10::TensorType>()
                                                   ->scalarType();

            const bool with_update = num_user_inputs == 3;
            const ScatterKind key{reduction, input_type, with_update};
            scatters[key].push_back(user);
          }
        }
      }
    }
  };

  while (!queue.empty()) {
    auto *node = queue.front();
    queue.pop();
    const torch::jit::Symbol kind = node->kind();

    // If scatter, push back.
    if (kind == symbols::popart::scatterreduce) {
      queue.push(node);
    } else {
      add_children_to_queue(node);
    }

    // If all elements of the queue are scatter.
    if (queue.size() == num_scatters_in_queue) {
      // Clear queue.
      queue = std::queue<torch::jit::Node *>();
      num_scatters_in_queue = 0;

      // Merge scatters that have been encountered twice.
      for (auto &&[scatter_kind, scatter_vec] : scatters) {
        if (scatter_vec.size() > 1) {
          const bool with_update = std::get<bool>(scatter_kind);
          const auto &merged_scatters =
              dispatchScatters(graph, scatter_vec, with_update);
          for (torch::jit::Node *scatter_node : merged_scatters) {
            add_children_to_queue(scatter_node);
          }
        } else {
          add_children_to_queue(scatter_vec.front());
        }
      }
      scatters.clear();
    }
  }
}

} // namespace poptorch
