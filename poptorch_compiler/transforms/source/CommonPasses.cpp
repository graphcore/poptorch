// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "passes/CommonPasses.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

namespace poptorch_ir {

void addOverwriteHandlingPasses(mlir::PassManager &manager) {
  manager.addPass(createOutplaceViewOpsPass());
  manager.addPass(createRemoveOverwritePass());
}

} // namespace poptorch_ir
