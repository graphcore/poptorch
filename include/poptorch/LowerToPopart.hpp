#ifndef INCLUDE_POPTORCH_LOWER_TO_POPART_H
#define INCLUDE_POPTORCH_LOWER_TO_POPART_H


#include <string>
#include <vector>

namespace torch {
namespace jit {
class Graph;
} // namespace jit
} // namespace torch

namespace poptorch {

using InputTensorType = std::vector<std::pair<std::string, std::vector<int64_t>>>;


/*
 * Take the transformed graph and create a poponnx graph from it.
 */
extern void lowerToPopart(torch::jit::Graph &graph, InputTensorType& in_tensors);

} // namespace Poptorch

#endif // INCLUDE_POPTORCH_LOWER_TO_POPART_H
