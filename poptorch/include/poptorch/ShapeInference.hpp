// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_SHAPE_INFERENCE_H
#define INCLUDE_POPTORCH_SHAPE_INFERENCE_H
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/script.h>

namespace poptorch {

void propagateInputShapes(torch::jit::Graph *graph);

}

#endif
