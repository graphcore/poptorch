#ifndef INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_
#define INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_


namespace torch {
namespace jit {
class Graph;
} // namespace jit
} // namespace torch


namespace poptorch {

void Canonicalize(torch::jit::Graph& graph);

}


#endif // INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_