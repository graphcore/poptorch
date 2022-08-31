// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_ALIAS_PROCESSING_H
#define INCLUDE_POPTORCH_ALIAS_PROCESSING_H

namespace torch {
namespace jit {
struct Graph;
} // namespace jit
} // namespace torch

namespace poptorch {

// Remove instances of aten::alias in the graph by replacing the outputs with
// the original (aliased) output. The known source of aliases is when an
// operation takes place on a wrapped buffer, for which the return value tensor
// is aliased and then set to be a member of the original (wrapper) subclass.
void resolveAliases(torch::jit::Graph *graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_ALIAS_PROCESSING_H
