// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOWER_TO_POPART_H
#define INCLUDE_POPTORCH_LOWER_TO_POPART_H

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "poptorch/PoplarExecutable.hpp"
#include "poptorch/SessionOptionsParser.hpp"

namespace poptorch {
namespace popart_compiler {
class SessionOptions;
}

namespace detail {
class LowerToPopartImpl;
} // namespace detail

// CallbackMetadata is used to pass information from python to the poplar custom
// op for CPU ops. The string is the ID given by the user to each op.
using CPUCallbackMap =
    std::unordered_map<std::string, popart_compiler::CallbackMetadata>;

struct Anchor {
  Anchor(std::string n, std::uint8_t m, size_t p)
      : name(std::move(n)), mode(m), period(p) {}

  std::string name;
  std::uint8_t mode;
  size_t period;
};

using AnchorList = std::vector<Anchor>;

/*
 * Take the transformed graph and create a poponnx graph from it.
 */

struct InplaceGraphInfo;

class LowerToPopart {
public:
  LowerToPopart(torch::jit::Graph *graph, InplaceGraphInfo &&inplace_info,
                bool training, std::vector<popart_compiler::Optimizer> &&opt,
                const popart_compiler::SessionOptions &options,
                const AttributeAccessor &attribute_accessor,
                CPUCallbackMap callback, AnchorList &&anchors);
  LowerToPopart(LowerToPopart &&lower);
  ~LowerToPopart();

  void lower();
  std::shared_ptr<poptorch::PoplarExecutable> compile();
  std::shared_ptr<poptorch::PoplarExecutable>
  loadExecutableFromFile(const std::string &input_filename);

private:
  std::unique_ptr<detail::LowerToPopartImpl> _impl;
};

} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOWER_TO_POPART_H
