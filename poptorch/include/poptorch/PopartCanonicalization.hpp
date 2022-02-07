// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_
#define INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_

#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
struct Graph;
} // namespace jit
} // namespace torch

namespace at {
class Tensor;
} // namespace at

namespace poptorch {
/*
   The first canonicalization pass cleans up the pytorch IR to use popart
   specific operations and will remove all others. Constants will be folded into
   the attributes of the ops themselves.
*/
void canonicalize(torch::jit::Graph *graph);

/*
 * The second late canonicalization pass will take the popart code and will
 * enforce any constraints that aren't fixed by popart itself.
 */
void canonicalizeLate(torch::jit::Graph *graph);

void canonicalizeLists(torch::jit::Graph *graph);

/*
 * Warn if any Aten ops remain in the graph after we have run canonicalisation
 * so the user can report exactly what operation we are missing.
 */
void warnOnUnsupportedAten(torch::jit::Graph *graph);

void annotateSubgraphs(torch::jit::Graph *graph, bool training);

void removeSurplusIdentityLosses(torch::jit::Graph *graph);

void canonicaliseHalfInputs(torch::jit::Graph *graph,
                            const std::vector<at::Tensor> &in_tensors,
                            const std::vector<at::Tensor> &parameters);

// Resolve types which are ambiguiously between half or float.
void resolveHalfOrFloat(torch::jit::Graph *graph);

// Clean up the graph if it is using CPU offloading.
void cpuOffloadingCleanup(torch::jit::Graph *graph);

// Handle the 'requires_grad=False' flag on tensors.
void addDetachOperations(torch::jit::Graph *graph);

// Popart scatterreduceop allows for non-expanded index to be passed in. It is
// essentially a fused and more efficient version of the expand + scatterreduce.
// This pass identifies all of the valid optimization cases and removes the
// explicit and sub-optimal index expansion before the scatter_add ops.
void removeScatterAddIndexExpansion(torch::jit::Graph *graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_
