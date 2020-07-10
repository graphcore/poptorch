// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_PEEPHOLE_H
#define INCLUDE_POPTORCH_PEEPHOLE_H
#include <torch/script.h>

namespace poptorch {

void peepholeOptimizations(torch::jit::Graph *graph, bool training);

} // namespace poptorch

#endif
