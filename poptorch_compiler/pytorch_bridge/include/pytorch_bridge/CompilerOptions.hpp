// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_OPTIONS_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_OPTIONS_HPP_

#include <vector>

namespace poptorch {

struct CompilerOptions {
  struct Dispatcher {
    // NOTE: std::string-s are avoided here due to ABI issues
    std::vector<std::vector<char>> source_location_excludes;
    bool check_added_ops = true;
  };
  Dispatcher dispatcher;
};

} // namespace poptorch

#endif
