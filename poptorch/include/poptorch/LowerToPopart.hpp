// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOWER_TO_POPART_H
#define INCLUDE_POPTORCH_LOWER_TO_POPART_H

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/PopartEnums.hpp"
#include "poptorch/PoplarExecutable.hpp"

namespace pybind11 {
class function;
}
namespace py = pybind11;

namespace poptorch {
class SessionOptions;

/*
 * Take the transformed graph and create a poponnx graph from it.
 */
std::shared_ptr<poptorch::PoplarExecutable>
lowerToPopart(torch::jit::Graph *graph, std::vector<at::Tensor> *in_tensors,
              std::vector<at::Tensor> parameters,
              std::vector<std::string> parameter_names, bool training,
              std::vector<Optimizer> &&opt, const SessionOptions &options,
              const py::function &attribute_accessor);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOWER_TO_POPART_H
