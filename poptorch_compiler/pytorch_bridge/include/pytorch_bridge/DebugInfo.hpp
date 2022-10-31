// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_DEBUG_INFO_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_DEBUG_INFO_HPP_

#include <memory>
#include <vector>

namespace poptorch_ir {

struct GraphDebugInfo {
  // Note these are shared with the tensor details
  std::shared_ptr<std::vector<char>> initial_graph;
  std::shared_ptr<std::vector<char>> cached_graph;
};

struct TensorDebugInfo {
  GraphDebugInfo debug_info;
  std::size_t output_idx;
};

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_DEBUG_INFO_HPP_
