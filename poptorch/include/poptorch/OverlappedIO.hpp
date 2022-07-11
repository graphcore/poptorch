// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_OVERLAPPED_IO_H
#define INCLUDE_POPTORCH_OVERLAPPED_IO_H

namespace torch {
namespace jit {
struct Graph;

} // namespace jit
} // namespace torch

namespace poptorch {

// TODO(T64770): Remove verify_only=false path when removing JIT tracing
// Turns any set_overlap_for_input nodes applied to inputs into attributes of
// the parameter node. These attributes specify any host IO Overlapped for the
// input
void attributiseOverlappedIO(torch::jit::Graph *graph,
                             bool verify_only = false);

// In the dispatcher, the attributes are set beforehand during dispatch.
// This pass verifies that those attributes were set correctly for each node,
// and removes them from the graph
void verifyOverlappedIOForDispatch(torch::jit::Graph *graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_OVERLAPPED_IO_H
