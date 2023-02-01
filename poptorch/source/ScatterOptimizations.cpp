// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <array>
#include <map>
#include <queue>
#include <set>
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
torch::jit::Node *
createGroupedScatterReduceNode(torch::jit::Graph *graph,
                               const torch::jit::node_list &scatter_nodes,
                               const std::vector<torch::jit::Value *> &inputs) {
  const int64_t num_groups = scatter_nodes.size();
  auto *const node_with_attributes = scatter_nodes.back();
  const auto axis_size =
      node_with_attributes->i(c10::Symbol::attr("axis_size"));
  const auto old_axis = node_with_attributes->i(c10::Symbol::attr("axis"));
  const auto reduction =
      node_with_attributes->i(c10::Symbol::attr("reduction"));
  return createGroupedscatterreduce(graph, inputs, axis_size, old_axis + 1,
                                    reduction, num_groups);
}

using ScatterInputArgs =
    std::tuple<torch::jit::Value *, torch::jit::Value *, torch::jit::Value *>;
using GroupedInputArgs = std::array<std::vector<torch::jit::Value *>, 3>;

ScatterInputArgs getScatterInputArgs(const torch::jit::Node *scatter_node,
                                     bool with_update) {
  return {scatter_node->input(0), scatter_node->input(1),
          (with_update ? scatter_node->input(2) : nullptr)};
}

torch::jit::node_list removeDuplicates(const torch::jit::node_list &scatters,
                                       bool with_update) {
  std::map<ScatterInputArgs, torch::jit::Node *> input_args_to_scatter_nodes;
  std::unordered_set<torch::jit::Node *> to_destroy;

  for (torch::jit::Node *scatter_node : scatters) {
    const auto scatter_node_inputs =
        getScatterInputArgs(scatter_node, with_update);
    auto stored_scatter_node_it =
        input_args_to_scatter_nodes.find(scatter_node_inputs);
    const bool is_duplicate =
        stored_scatter_node_it != input_args_to_scatter_nodes.end();
    if (is_duplicate) {
      auto *const stored_scatter_node = stored_scatter_node_it->second;
      replaceOutputUse(scatter_node->output(), stored_scatter_node->output());
      to_destroy.insert(scatter_node);
    } else {
      input_args_to_scatter_nodes.emplace(scatter_node_inputs, scatter_node);
    }
  }

  searchAndPossiblyDestroy(to_destroy);

  torch::jit::node_list unique_scatter_nodes(input_args_to_scatter_nodes.size(),
                                             nullptr);
  std::transform(
      input_args_to_scatter_nodes.begin(), input_args_to_scatter_nodes.end(),
      unique_scatter_nodes.begin(), [&](const auto &input_args_to_node) {
        return input_args_to_node.second;
      });

  return unique_scatter_nodes;
}

void sortInTopologicalOrder(torch::jit::node_list &nodes) {
  std::sort(nodes.begin(), nodes.end(),
            [=](const torch::jit::Node *lhs, const torch::jit::Node *rhs) {
              return lhs->isBefore(rhs);
            });
}

void moveOutputNodesAfterInsertionPoint(
    const torch::jit::node_list &scatter_nodes,
    torch::jit::Node *insertion_point_node) {

  const auto collect_output_nodes = [](const torch::jit::Node *node_to_process,
                                       std::queue<torch::jit::Node *> &queue) {
    for (const auto *output : node_to_process->outputs()) {
      for (const auto &use : output->uses()) {
        const auto &user = use.user;
        queue.push(user);
      }
    }
  };

  auto *tmp_insertion_point_node = insertion_point_node;

  for (torch::jit::Node *scatter : scatter_nodes) {

    if (scatter == insertion_point_node) {
      continue;
    }

    std::queue<torch::jit::Node *> nodes_to_move;
    collect_output_nodes(scatter, nodes_to_move);
    while (!nodes_to_move.empty()) {
      torch::jit::Node *node = nodes_to_move.front();
      nodes_to_move.pop();
      if (node->isBefore(tmp_insertion_point_node)) {
        node->moveAfter(tmp_insertion_point_node);
        tmp_insertion_point_node = node;
        collect_output_nodes(node, nodes_to_move);
      }
    }
  }
}

GroupedInputArgs groupScatterInputs(const torch::jit::node_list &scatter_nodes,
                                    bool with_update) {
  const int64_t num_groups = scatter_nodes.size();

  GroupedInputArgs grouped_input_nodes;
  for (auto &input_vec : grouped_input_nodes) {
    input_vec = std::vector<torch::jit::Value *>(num_groups, nullptr);
  }

  for (int64_t group_id = 0; group_id < num_groups; ++group_id) {

    std::tie(grouped_input_nodes[0][group_id], grouped_input_nodes[1][group_id],
             grouped_input_nodes[2][group_id]) =
        getScatterInputArgs(scatter_nodes[group_id], with_update);
  }

  return grouped_input_nodes;
}

std::vector<torch::jit::Value *>
concatGroupedInputs(torch::jit::Graph *graph, GroupedInputArgs &grouped_inputs,
                    bool with_update) {

  const std::size_t num_groups = grouped_inputs[0].size();

  for (std::size_t group_id = 0; group_id < num_groups; group_id++) {
    auto &src_input = grouped_inputs[0][group_id];
    src_input = createUnsqueeze(graph, {src_input}, {0})->output();

    auto &index_input = grouped_inputs[1][group_id];
    index_input = createUnsqueeze(graph, {index_input}, {0})->output();

    if (with_update) {
      auto &self_input = grouped_inputs[2][group_id];
      self_input = createUnsqueeze(graph, {self_input}, {0})->output();
    }
  }

  std::vector<torch::jit::Value *> grouped_scatter_args;
  grouped_scatter_args.reserve(3);

  grouped_scatter_args.push_back(
      createConcat(graph, grouped_inputs[0], 0)->output());
  grouped_scatter_args.push_back(
      createConcat(graph, grouped_inputs[1], 0)->output());

  if (with_update) {
    grouped_scatter_args.push_back(
        createConcat(graph, grouped_inputs[2], 0)->output());
  }

  return grouped_scatter_args;
}

void unpackGroupedScatterReduceOutputs(
    torch::jit::Graph *graph, torch::jit::Node *grouped_scatter_reduce_node,
    const torch::jit::node_list &fused_scatter_nodes) {
  std::unordered_set<torch::jit::Node *> to_destroy;
  const int64_t num_groups = fused_scatter_nodes.size();

  for (int64_t group_id = 0; group_id < num_groups; ++group_id) {
    auto *slice = createSlice(graph, {grouped_scatter_reduce_node->output()},
                              {group_id + 1}, {group_id}, {0});
    auto *squeeze = createSqueeze(graph, {slice->output()}, {0});

    // Replace outputs with grouped version.
    auto *scatter_to_replace = fused_scatter_nodes[group_id];
    for (auto *output : scatter_to_replace->outputs()) {
      replaceOutputUse(output, squeeze->output());
    }
    to_destroy.insert(scatter_to_replace);
  }
  // Destroy merged scatters.
  searchAndPossiblyDestroy(to_destroy);
}

torch::jit::Node *
mergeScatterReduceNodes(torch::jit::Graph *graph,
                        const torch::jit::node_list &scatter_nodes,
                        bool with_update) {

  torch::jit::node_list unique_scatter_nodes =
      poptorch::removeDuplicates(scatter_nodes, with_update);
  sortInTopologicalOrder(unique_scatter_nodes);

  torch::jit::Node *insertion_point_node = unique_scatter_nodes.back();
  moveOutputNodesAfterInsertionPoint(unique_scatter_nodes,
                                     insertion_point_node);

  auto grouped_inputs = groupScatterInputs(unique_scatter_nodes, with_update);

  const WithNodeMetadata meta{insertion_point_node};
  const torch::jit::WithInsertPoint insertion_point(insertion_point_node);

  const auto grouped_scatter_reduce_args =
      concatGroupedInputs(graph, grouped_inputs, with_update);
  auto *grouped_scatter_reduce_node = createGroupedScatterReduceNode(
      graph, unique_scatter_nodes, grouped_scatter_reduce_args);

  unpackGroupedScatterReduceOutputs(graph, grouped_scatter_reduce_node,
                                    unique_scatter_nodes);

  return grouped_scatter_reduce_node;
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
    UNUSED(_);

    if (merge_candidates.size() > 1) {
      grouped_scatters.push_back(
          mergeScatterReduceNodes(graph, merge_candidates, with_update));
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
