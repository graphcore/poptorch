#ifndef INCLUDE_POPTORCH_LOWER_TO_POPART_H
#define INCLUDE_POPTORCH_LOWER_TO_POPART_H

#include <torch/csrc/jit/ir.h>

#include <string>
#include <vector>

namespace poptorch {

/*
 * Take the transformed graph and create a poponnx graph from it.
 */
at::IValue lowerToPopart(torch::jit::Graph &graph, std::vector<at::Tensor>& inTensors);

} // namespace Poptorch

#endif // INCLUDE_POPTORCH_LOWER_TO_POPART_H
