#ifndef INCLUDE_POPTORCH_LOWER_TO_POPART_H
#define INCLUDE_POPTORCH_LOWER_TO_POPART_H

#include <poptorch/PoplarExecutable.hpp>
#include <string>
#include <torch/csrc/jit/ir.h>
#include <vector>

namespace poptorch {

/*
 * Take the transformed graph and create a poponnx graph from it.
 */
std::shared_ptr<poptorch::PoplarExecutable>
lowerToPopart(torch::jit::Graph &graph, std::vector<at::Tensor> &inTensors,
              std::vector<at::Tensor> &parameters, std::uint64_t steps, bool training,
	      std::uint64_t replicationFactor,  std::uint64_t gradientAccumulation);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOWER_TO_POPART_H
