// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_OVERLAPPED_IO_H
#define INCLUDE_POPTORCH_OVERLAPPED_IO_H

namespace torch {
namespace jit {
struct Graph;

} // namespace jit
} // namespace torch

namespace poptorch {

// Turns any set_overlap_for_input nodes applied to inputs into attributes of
// the parameter node. These attributes specify any host IO Overlapped for the
// input
void attributiseOverlappedIO(torch::jit::Graph *graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_OVERLAPPED_IO_H
