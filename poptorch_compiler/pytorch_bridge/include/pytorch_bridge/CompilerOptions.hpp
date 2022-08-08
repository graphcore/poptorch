// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_OPTIONS_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_OPTIONS_HPP_

#include <vector>

namespace poptorch {

struct CompilerOptions {
  static CompilerOptions eagerOptions() {
    CompilerOptions opts;
    opts.eager.eager_mode = true;
    return opts;
  }

  struct Dispatcher {
    // NOTE: std::string-s are avoided here due to ABI issues
    std::vector<std::vector<char>> source_location_excludes;
  };
  struct Eager {
    bool eager_mode = false;
    bool use_lazy_tensor = false;
  };

  Dispatcher dispatcher;
  Eager eager;
};

} // namespace poptorch

#endif