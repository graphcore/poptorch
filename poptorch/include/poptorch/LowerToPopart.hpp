// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOWER_TO_POPART_H
#define INCLUDE_POPTORCH_LOWER_TO_POPART_H

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "poptorch/PoplarExecutable.hpp"

namespace poptorch {

/*
 * Take the transformed graph and create a poponnx graph from it.
 */
std::shared_ptr<poptorch::PoplarExecutable> lowerToPopart(
    torch::jit::Graph &graph, std::vector<at::Tensor> &inTensors,
    std::vector<at::Tensor> &parameters, std::uint64_t steps, bool training,
    std::uint64_t replicationFactor, std::uint64_t gradientAccumulation,
    const std::unordered_map<std::string, std::pair<float, bool>> &opt,
    bool profile);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOWER_TO_POPART_H
