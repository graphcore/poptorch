// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <queue>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>
#include <unordered_set>
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
    auto expanded_index_shape = shapeFromTensor(index);

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
  auto axis_size = node_with_attributes->i(c10::Symbol::attr("axis_size"));
  auto old_axis = node_with_attributes->i(c10::Symbol::attr("axis"));
  auto reduction = node_with_attributes->i(c10::Symbol::attr("reduction"));
  return createGroupedscatterreduce(graph, inputs, axis_size, old_axis + 1,
                                    reduction, num_scatters);
}

torch::jit::Node *mergeSet(torch::jit::Graph *graph,
                           torch::jit::node_list &scatters) {
  std::unordered_map<torch::jit::Value *, torch::jit::Node *> srcs;
  std::unordered_set<torch::jit::Node *> to_destroy;

  bool with_update = scatters[0]->inputs().size() == 3;
  auto *last_scatter = scatters[0];
  srcs.insert({last_scatter->input(0), last_scatter});
  for (size_t i = 1; i < scatters.size(); i++) {
    auto *scatter = scatters[i];
    // If encountered, remove the repetition.
    if (srcs.find(scatter->input(0)) != srcs.end()) {
      // Check with second input.
      auto *scatter_with_same_input0 = srcs[scatter->input(0)];
      bool is_repetition =
          scatter->input(1) == scatter_with_same_input0->input(1);
      if (with_update) {
        is_repetition = is_repetition &&
                        scatter->input(2) == scatter_with_same_input0->input(2);
      }

      if (is_repetition) {
        replaceOutputUse(scatter->output(), srcs[scatter->input(0)]->output());
        to_destroy.insert(scatter);
        scatters.erase(scatters.begin() + i);
        i--;
      }
    } else {
      srcs.insert({scatter->input(0), scatter});
      // Find last scatter to execute. This should be insert point for grouped
      // scatter.
      if (last_scatter->isBefore(scatter)) {
        last_scatter = scatter;
      }
    }
  }

  auto insert_outputs_users = [](torch::jit::Node *node_to_process,
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
  auto *concat0 = createConcat(graph, inputs0_scat, 0)->output();
  auto *concat1 = createConcat(graph, inputs1_scat, 0)->output();
  torch::jit::Value *concat2;
  if (with_update) {
    concat2 = createConcat(graph, inputs2_scat, 0)->output();
  }

  std::vector<torch::jit::Value *> args{concat0, concat1};
  if (with_update) {
    args.push_back(concat2);
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
                                       torch::jit::node_list &scatters) {

  // Separate the scatters by their shape (converted to string).
  std::unordered_map<std::string, torch::jit::node_list> sets_by_ind;
  for (auto *scatter : scatters) {
    auto ind_shape = shapeFromTensor(scatter->input(1));

    std::ostringstream shape_str_stream;
    std::copy(ind_shape.begin(), ind_shape.end(),
              std::ostream_iterator<int>(shape_str_stream, " "));
    const std::string &shape_str = shape_str_stream.str();

    if (sets_by_ind.find(shape_str) == sets_by_ind.end()) {
      sets_by_ind.insert({shape_str, {scatter}});
    } else {
      sets_by_ind[shape_str].push_back(scatter);
    }
  }

  torch::jit::node_list grouped_scatters;
  for (auto &it_by_ind : sets_by_ind) {
    auto &set_by_ind = it_by_ind.second;

    // Separate the scatters by their axis attribute.
    std::unordered_map<int, torch::jit::node_list> sets_by_axis;
    for (auto *scatter_by_ind : set_by_ind) {
      const auto axis = scatter_by_ind->i(c10::Symbol::attr("axis"));
      if (sets_by_axis.find(axis) == sets_by_axis.end()) {
        sets_by_axis.insert({axis, {scatter_by_ind}});
      } else {
        sets_by_axis[axis].push_back(scatter_by_ind);
      }
    }

    // Merge sets.
    for (auto &it_by_axis : sets_by_axis) {
      auto &set_by_axis = it_by_axis.second;
      if (set_by_axis.size() > 1) {
        auto *grouped_scatter = mergeSet(graph, set_by_axis);
        grouped_scatters.push_back(grouped_scatter);
      } else {
        grouped_scatters.push_back(set_by_axis[0]);
      }
    }
  }
  // Return grouped scatters.
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
  constexpr int num_reductions = 6; // add, min, max, mul, none, mean

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
  std::unordered_map<torch::jit::Node *, std::size_t> occurs;
  // The unordered_set elements mean that children have been added to the queue.
  // std::unordered_set<torch::jit::Node *> visited;
  // Divide the scatters according to the type of reduction, types and whether
  // an update is used.
  torch::jit::node_list scatters[num_reductions][at::NumScalarTypes]
                                [2 /*with_update*/];
  std::size_t num_scatters_in_queue = 0;

  // Lambda to add the children of the vertex.
  auto add_children_to_queue = [&](torch::jit::Node *node) {
    for (auto *output : node->outputs()) {
      for (const auto &use : output->uses()) {
        const auto &user = use.user;
        if (occurs.find(user) != occurs.end()) {
          occurs[user]++;
        } else {
          occurs[user] = 1;
        }

        if (occurs[user] == user->inputs().size()) {
          queue.push(user);
          const torch::jit::Symbol user_kind = user->kind();
          if (user_kind == symbols::popart::scatterreduce) {
            num_scatters_in_queue++;
            const auto reduction = user->i(c10::Symbol::attr("reduction"));
            const auto input_type = *user->input(0)
                                         ->type()
                                         ->expect<c10::TensorType>()
                                         ->scalarType();
            // ScalarType is enum of int8_t.
            const auto input_type_idx = static_cast<int8_t>(input_type);
            const int with_update =
                static_cast<int>(user->inputs().size() == 3);
            scatters[reduction][input_type_idx][with_update].push_back(user);
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
      for (auto &scatter_set_by_reduction : scatters) {
        for (auto &scatter_set_by_type : scatter_set_by_reduction) {
          for (auto &scatter_set : scatter_set_by_type) {
            if (scatter_set.size() > 1) {
              const auto &grouped_scatters =
                  dispatchScatters(graph, scatter_set);
              for (auto *grouped_scatter : grouped_scatters) {
                add_children_to_queue(grouped_scatter);
              }
            } else if (scatter_set.size() == 1) {
              add_children_to_queue(scatter_set[0]);
            }
            scatter_set.clear();
          }
        }
      }
    }
  }
}

} // namespace poptorch
