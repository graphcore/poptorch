// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
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

namespace {

using InputArgs =
    std::tuple<torch::jit::Value *, torch::jit::Value *, torch::jit::Value *>;
using GroupedInputArgs = std::array<std::vector<torch::jit::Value *>, 3>;
using GroupedOpFactory = std::function<torch::jit::Node *(
    torch::jit::Graph *, const torch::jit::node_list &,
    const std::vector<torch::jit::Value *> &)>;

void groupScatterReduceNodes(torch::jit::Graph *graph);
void groupGatherNodes(torch::jit::Graph *graph);
void initQueue(torch::jit::Graph *graph, std::queue<torch::jit::Node *> &queue,
               torch::jit::node_list &barriers);
std::size_t deduceOpStage(const torch::jit::Node *node,
                          const torch::jit::node_list &barriers);
std::vector<torch::jit::Value *>
concatGroupedInputs(torch::jit::Graph *graph, GroupedInputArgs &grouped_inputs,
                    bool with_update);
torch::jit::Node *
createGroupedScatterReduceNode(torch::jit::Graph *graph,
                               const torch::jit::node_list &scatter_nodes,
                               const std::vector<torch::jit::Value *> &inputs);
torch::jit::Node *
createGroupedGatherNode(torch::jit::Graph *graph,
                        const torch::jit::node_list &gather_nodes,
                        const std::vector<torch::jit::Value *> &inputs);
torch::jit::node_list dispatch(torch::jit::Graph *graph,
                               torch::jit::node_list &nodes,
                               const GroupedOpFactory &createGroupedOpFn,
                               bool with_update = false);
InputArgs getInputArgs(const torch::jit::Node *node, bool with_update);
GroupedInputArgs groupInputs(const torch::jit::node_list &nodes,
                             bool with_update);
torch::jit::Node *mergeNodes(torch::jit::Graph *graph,
                             const torch::jit::node_list &nodes,
                             const GroupedOpFactory &createGroupedOpFn,
                             bool with_update);
void moveOutputNodesAfterInsertionPoint(const torch::jit::node_list &nodes,
                                        torch::jit::Node *insertion_point_node);
torch::jit::node_list removeDuplicates(const torch::jit::node_list &nodes,
                                       bool with_update);
void sortInTopologicalOrder(torch::jit::node_list &nodes);
void unpackGroupedOutputs(torch::jit::Graph *graph,
                          torch::jit::Node *grouped_node,
                          const torch::jit::node_list &fused_nodes);

} // namespace

/*
 * Algorithm:
 * 1. Move the BFS around the graph and add only those that all inputs are
 *    encountered until the entire queue is scatters and gathers.
 * 2. Merge the scatters and gathers.
 * 3. Add outputs to queue and remove scatters and gathers.
 * 4. If queue is not empty go to step 1.
 */
void groupScatterReduceAndGatherNodes(torch::jit::Graph *graph) {
  groupScatterReduceNodes(graph);
  // Enable after fix AFS-197
  // groupGatherNodes(graph);
}

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
    node->i_(c10::Symbol::attr("enable_index_broadcast"), 1);

    to_delete.push_back(index_producer);
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

namespace {
void groupScatterReduceNodes(torch::jit::Graph *graph) {
  logging::LogContext const ctx{"groupScatterReduceNodes"};

  // Queue contains fully reached nodes.
  std::queue<torch::jit::Node *> queue;
  torch::jit::node_list barriers;
  initQueue(graph, queue, barriers);

  // The unordered_map elements represent the number of times the node was
  // reached.
  std::unordered_map<torch::jit::Node *, std::size_t> node_num_visited_inputs;
  // The unordered_set elements mean that children have been added to the queue.

  static constexpr auto with_update_idx = 3;

  using ScatterKind =
      std::tuple<std::int64_t /*reduction*/, at::ScalarType /*input_type*/,
                 bool /*index_broadcast_enabled*/, bool /*with_update*/,
                 std::size_t /*stage*/>;
  std::map<ScatterKind, torch::jit::node_list> scatters;

  std::size_t optimization_candidates = 0;

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
            ++optimization_candidates;
            const std::int64_t reduction =
                user->i(c10::Symbol::attr("reduction"));
            const at::ScalarType input_type = *user->input(0)
                                                   ->type()
                                                   ->expect<c10::TensorType>()
                                                   ->scalarType();

            const bool with_update = num_user_inputs == 3;
            const bool index_broadcast_enabled =
                user->i(c10::Symbol::attr("enable_index_broadcast")) != 0;
            const std::size_t stage = deduceOpStage(user, barriers);
            const ScatterKind key{reduction, input_type,
                                  index_broadcast_enabled, with_update, stage};
            scatters[key].push_back(user);
          }
        }
      }
    }
  };

  const auto merge_scatters = [&]() {
    for (auto &&[scatter_kind, scatter_vec] : scatters) {
      if (scatter_vec.size() > 1) {
        const bool with_update = std::get<with_update_idx>(scatter_kind);
        const auto &merged_scatters = dispatch(
            graph, scatter_vec, createGroupedScatterReduceNode, with_update);
        for (torch::jit::Node *scatter_node : merged_scatters) {
          add_children_to_queue(scatter_node);
        }
      } else {
        add_children_to_queue(scatter_vec.front());
      }
    }
    scatters.clear();
  };

  while (!queue.empty()) {
    auto *node = queue.front();
    queue.pop();
    const torch::jit::Symbol kind = node->kind();

    // If scatter or gather, push back.
    if (kind == symbols::popart::scatterreduce) {
      queue.push(node);
    } else {
      add_children_to_queue(node);
    }

    // If all elements of the queue are scatters and gathers.
    if (queue.size() == optimization_candidates) {
      // Clear queue.
      queue = std::queue<torch::jit::Node *>();
      optimization_candidates = 0;
      // Merge scatters and gathers that have been encountered twice.
      merge_scatters();
    }
  }
}

[[maybe_unused]] void groupGatherNodes(torch::jit::Graph *graph) {
  logging::LogContext const ctx{"groupGatherNodes"};

  // Queue contains fully reached nodes.
  std::queue<torch::jit::Node *> queue;
  torch::jit::node_list barriers;
  initQueue(graph, queue, barriers);

  // The unordered_map elements represent the number of times the node was
  // reached.
  std::unordered_map<torch::jit::Node *, std::size_t> node_num_visited_inputs;
  // The unordered_set elements mean that children have been added to the queue.

  using GatherKind =
      std::tuple<at::ScalarType /*input_type*/, std::size_t /*stage*/>;
  std::map<GatherKind, torch::jit::node_list> gathers;

  std::size_t optimization_candidates = 0;

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
          if (user->kind() == symbols::popart::gather) {
            ++optimization_candidates;
            const at::ScalarType input_type = *user->input(0)
                                                   ->type()
                                                   ->expect<c10::TensorType>()
                                                   ->scalarType();

            const std::size_t stage = deduceOpStage(user, barriers);
            const GatherKind key{input_type, stage};
            gathers[key].push_back(user);
          }
        }
      }
    }
  };

  const auto merge_gathers = [&]() {
    for (auto &&[_, gather_vec] : gathers) {
      UNUSED(_);
      if (gather_vec.size() > 1) {
        const auto &merged_gathers =
            dispatch(graph, gather_vec, createGroupedGatherNode);
        for (torch::jit::Node *gather_node : merged_gathers) {
          add_children_to_queue(gather_node);
        }
      } else {
        add_children_to_queue(gather_vec.front());
      }
    }
    gathers.clear();
  };

  while (!queue.empty()) {
    auto *node = queue.front();
    queue.pop();
    const torch::jit::Symbol kind = node->kind();

    // If scatter or gather, push back.
    if (kind == symbols::popart::gather) {
      queue.push(node);
    } else {
      add_children_to_queue(node);
    }

    // If all elements of the queue are scatters and gathers.
    if (queue.size() == optimization_candidates) {
      // Clear queue.
      queue = std::queue<torch::jit::Node *>();
      optimization_candidates = 0;
      // Merge scatters and gathers that have been encountered twice.
      merge_gathers();
    }
  }
}

void initQueue(torch::jit::Graph *graph, std::queue<torch::jit::Node *> &queue,
               torch::jit::node_list &barriers) {
  // Add roots to queue.
  std::unordered_set<torch::jit::Node *> added;
  for (torch::jit::Node *node : graph->nodes()) {
    if (node->inputs().empty()) {
      if (added.find(node) == added.end()) {
        queue.push(node);
        added.insert(node);
      }
    }
    if (node->kind() == symbols::poptorch::begin_ipu_block) {
      barriers.push_back(node);
    }
  }
  for (torch::jit::Value *input : graph->inputs()) {
    auto *node = input->node();
    if (added.find(node) == added.end()) {
      queue.push(node);
      added.insert(node);
    }
  }
}

// Find which phase the fused operation is in
std::size_t deduceOpStage(const torch::jit::Node *node,
                          const torch::jit::node_list &barriers) {
  std::size_t stage = 0;
  while (stage < barriers.size() && !node->isBefore(barriers[stage])) {
    stage++;
  }
  return stage;
}

torch::jit::node_list dispatch(torch::jit::Graph *graph,
                               torch::jit::node_list &nodes,
                               const GroupedOpFactory &createGroupedOpFn,
                               bool with_update) {
  using Shape = std::vector<std::int64_t /*dim*/>;
  using Group = std::tuple<std::int64_t /*axis*/, Shape /*index*/,
                           Shape /*src*/, Shape /*self*/>;

  std::map<Group, torch::jit::node_list> group_to_merge_candidates;

  for (torch::jit::Node *node : nodes) {
    const std::int64_t axis = node->i(c10::Symbol::attr("axis"));
    const Shape src_shape = shapeFromTensor(node->input(0));
    const Shape index_shape = shapeFromTensor(node->input(1));
    const Shape self_shape =
        with_update ? shapeFromTensor(node->input(2)) : Shape{};

    const Group key{axis, index_shape, src_shape, self_shape};
    group_to_merge_candidates[key].push_back(node);
  }

  torch::jit::node_list grouped_nodes;
  for (auto &&[_, merge_candidates] : group_to_merge_candidates) {
    UNUSED(_);

    if (merge_candidates.size() > 1) {
      grouped_nodes.push_back(
          mergeNodes(graph, merge_candidates, createGroupedOpFn, with_update));
    } else {
      grouped_nodes.push_back(merge_candidates.front());
    }
  }

  return grouped_nodes;
}

torch::jit::Node *mergeNodes(torch::jit::Graph *graph,
                             const torch::jit::node_list &nodes,
                             const GroupedOpFactory &createGroupedOpFn,
                             bool with_update) {

  torch::jit::node_list unique_nodes =
      poptorch::removeDuplicates(nodes, with_update);
  sortInTopologicalOrder(unique_nodes);

  torch::jit::Node *insertion_point_node = unique_nodes.back();
  moveOutputNodesAfterInsertionPoint(unique_nodes, insertion_point_node);

  auto grouped_inputs = groupInputs(unique_nodes, with_update);

  const WithNodeMetadata meta{insertion_point_node};
  const torch::jit::WithInsertPoint insertion_point(insertion_point_node);

  const auto grouped_args =
      concatGroupedInputs(graph, grouped_inputs, with_update);
  torch::jit::Node *grouped_node;
  grouped_node = createGroupedOpFn(graph, unique_nodes, grouped_args);

  unpackGroupedOutputs(graph, grouped_node, unique_nodes);

  return grouped_node;
}

torch::jit::node_list removeDuplicates(const torch::jit::node_list &nodes,
                                       bool with_update) {
  std::map<InputArgs, torch::jit::Node *> input_args_to_nodes;
  std::unordered_set<torch::jit::Node *> to_destroy;

  for (torch::jit::Node *node : nodes) {
    const auto node_inputs = getInputArgs(node, with_update);
    auto stored_node_it = input_args_to_nodes.find(node_inputs);
    const bool is_duplicate = stored_node_it != input_args_to_nodes.end();
    if (is_duplicate) {
      auto *const stored_node = stored_node_it->second;
      replaceOutputUse(node->output(), stored_node->output());
      to_destroy.insert(node);
    } else {
      input_args_to_nodes.emplace(node_inputs, node);
    }
  }

  searchAndPossiblyDestroy(to_destroy);

  torch::jit::node_list unique_nodes(input_args_to_nodes.size(), nullptr);
  std::transform(input_args_to_nodes.begin(), input_args_to_nodes.end(),
                 unique_nodes.begin(), [&](const auto &input_args_to_node) {
                   return input_args_to_node.second;
                 });

  return unique_nodes;
}

void sortInTopologicalOrder(torch::jit::node_list &nodes) {
  std::sort(nodes.begin(), nodes.end(),
            [=](const torch::jit::Node *lhs, const torch::jit::Node *rhs) {
              return lhs->isBefore(rhs);
            });
}

void moveOutputNodesAfterInsertionPoint(
    const torch::jit::node_list &nodes,
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

  std::unordered_set<torch::jit::Node *> collected_nodes_to_move;
  for (torch::jit::Node *node : nodes) {
    if (node == insertion_point_node) {
      continue;
    }

    std::queue<torch::jit::Node *> nodes_to_move;
    collect_output_nodes(node, nodes_to_move);
    while (!nodes_to_move.empty()) {
      torch::jit::Node *node_to_move = nodes_to_move.front();
      nodes_to_move.pop();
      if (node_to_move->isBefore(insertion_point_node) &&
          collected_nodes_to_move.find(node_to_move) ==
              collected_nodes_to_move.end()) {
        collected_nodes_to_move.insert(node_to_move);
        collect_output_nodes(node_to_move, nodes_to_move);
      }
    }
  }

  torch::jit::node_list sorted_collected_nodes_to_move;
  sorted_collected_nodes_to_move.insert(sorted_collected_nodes_to_move.end(),
                                        collected_nodes_to_move.begin(),
                                        collected_nodes_to_move.end());
  sortInTopologicalOrder(sorted_collected_nodes_to_move);
  auto *tmp_insertion_point_node = insertion_point_node;
  for (auto *node_to_move : sorted_collected_nodes_to_move) {
    node_to_move->moveAfter(tmp_insertion_point_node);
    tmp_insertion_point_node = node_to_move;
  }
}

GroupedInputArgs groupInputs(const torch::jit::node_list &nodes,
                             bool with_update) {
  const int64_t num_groups = nodes.size();

  GroupedInputArgs grouped_input_nodes;
  for (auto &input_vec : grouped_input_nodes) {
    input_vec = std::vector<torch::jit::Value *>(num_groups, nullptr);
  }

  for (int64_t group_id = 0; group_id < num_groups; ++group_id) {

    std::tie(grouped_input_nodes[0][group_id], grouped_input_nodes[1][group_id],
             grouped_input_nodes[2][group_id]) =
        getInputArgs(nodes[group_id], with_update);
  }

  return grouped_input_nodes;
}

InputArgs getInputArgs(const torch::jit::Node *node, bool with_update) {
  return {node->input(0), node->input(1),
          (with_update ? node->input(2) : nullptr)};
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

  std::vector<torch::jit::Value *> grouped_args;
  grouped_args.reserve(3);

  grouped_args.push_back(createConcat(graph, grouped_inputs[0], 0)->output());
  grouped_args.push_back(createConcat(graph, grouped_inputs[1], 0)->output());

  if (with_update) {
    grouped_args.push_back(createConcat(graph, grouped_inputs[2], 0)->output());
  }

  return grouped_args;
}

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
  const bool enable_index_broadcast =
      node_with_attributes->i(c10::Symbol::attr("enable_index_broadcast")) != 0;

  return createGroupedscatterreduce(graph, inputs, axis_size, old_axis + 1,
                                    num_groups, enable_index_broadcast,
                                    reduction);
}

torch::jit::Node *
createGroupedGatherNode(torch::jit::Graph *graph,
                        const torch::jit::node_list &gather_nodes,
                        const std::vector<torch::jit::Value *> &inputs) {
  const int64_t num_groups = gather_nodes.size();
  auto *const node_with_attributes = gather_nodes.back();
  const auto axis = node_with_attributes->i(c10::Symbol::attr("axis"));
  return createGroupedgather(graph, inputs, axis + 1, num_groups);
}

void unpackGroupedOutputs(torch::jit::Graph *graph,
                          torch::jit::Node *grouped_node,
                          const torch::jit::node_list &fused_nodes) {
  std::unordered_set<torch::jit::Node *> to_destroy;
  const int64_t num_groups = fused_nodes.size();

  for (int64_t group_id = 0; group_id < num_groups; ++group_id) {
    torch::jit::Node *slice = createSlice(graph, {grouped_node->output()},
                                          {group_id + 1}, {group_id}, {0});
    torch::jit::Node *squeeze = createSqueeze(graph, {slice->output()}, {0});
    // Replace outputs with grouped version.
    torch::jit::Node *node_to_replace = fused_nodes[group_id];
    for (torch::jit::Value *output : node_to_replace->outputs()) {
      replaceOutputUse(output, squeeze->output());
    }
    to_destroy.insert(node_to_replace);
  }
  // Destroy merged scatters.
  searchAndPossiblyDestroy(to_destroy);
}

} // namespace

} // namespace poptorch
