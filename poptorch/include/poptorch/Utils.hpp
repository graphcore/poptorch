// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_UTILS_HPP
#define INCLUDE_POPTORCH_UTILS_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <string>

namespace poptorch {

torch::jit::Node *findEarliestUser(const torch::jit::Value *value);

std::string nodeToString(const torch::jit::Node *node);

std::string scalarTypeToOnnxString(at::ScalarType type);

// Delete a node and also its users if they are also unused.
void searchAndPossiblyDestroy(torch::jit::Node *node);

// Delete a node and also its users if they are also unused
// via an iterator
void searchAndPossiblyDestroy(torch::jit::graph_node_list_iterator *node_it);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_UTILS_HPP
