// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_REQUIRES_GRAD_H
#define INCLUDE_POPTORCH_REQUIRES_GRAD_H

namespace torch {
namespace jit {
struct Graph;
} // namespace jit
} // namespace torch

namespace poptorch {

// Autograd sets the requires_grad flag on the ATen tensors
// after we've instantiated the corresponding ATen node in the dispatcher.
// This pass goes through all the nodes in the ATen graph and sets the
// requires_graph flag on a node's outputs if any of its inputs has
// requires_grad set.
void fixRequiresGradFromDispatch(torch::jit::Graph *graph);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_REQUIRES_GRAD_H
