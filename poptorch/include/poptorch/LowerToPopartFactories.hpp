// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOWER_TO_POPART_FACTORIES_H
#define INCLUDE_POPTORCH_LOWER_TO_POPART_FACTORIES_H

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <vector>

#include "poptorch/LowerToPopart.hpp"
#include "poptorch/SessionOptionsParser.hpp"

namespace poptorch {

poptorch::LowerToPopart lowerToPopartFromDispatch(
    SessionOptionsParser &parser, bool training, AnchorList &&anchors_list,
    const std::function<void()> &initCallbackBuffers,
    std::vector<popart_compiler::Optimizer> &&optimizers,
    const AttributeAccessor &attribute_accessor, CPUCallbackMap &callbacks);
} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOWER_TO_POPART_FACTORIES_H
