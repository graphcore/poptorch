// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_
#define INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_

namespace torch {
namespace jit {
class Graph;
} // namespace jit
} // namespace torch

#include <string>
#include <unordered_map>

namespace poptorch {
/*
    The first canonicalization pass cleans up the pytorch IR to use popart
   specific operations and will remove all others. Constants will be folded into
   the attributes of the ops themselves.
*/
void Canonicalize(torch::jit::Graph &graph);

/*
 * The second late canonicalization pass will take the popart code and will
 * enforce any constraints that aren't fixed by popart itself.
 */
void CanonicalizeLate(torch::jit::Graph &graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_TRANSFORM_ATEN_TO_POPART_HPP_
