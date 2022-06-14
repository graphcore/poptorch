// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPIT_PASS_HPP_
#define POPTORCH_LOWER_TO_POPIT_PASS_HPP_

#include <memory>

namespace mlir {
template <typename T> class OperationPass;
class ModuleOp;
class Type;
} // namespace mlir

namespace popit {
class TensorSpec;
} // namespace popit

namespace poptorch_ir {

class PopitContext;

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerToPopitPass(PopitContext &context);

popit::TensorSpec getTensorSpec(mlir::Type mlirType);

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPIT_PASS_HPP_
