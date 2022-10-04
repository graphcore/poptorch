// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_
#define POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_

#include <llvm/ADT/DenseMap.h>
#include <memory>
#include <string>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace mlir {
class ModuleOp;
class TimingScope;
class Value;
class RankedTensorType;
} // namespace mlir

namespace poptorch_ir {

class PopitContext;
class NonRestartingMLIRTimer;

class PopitExecutor {
public:
  PopitExecutor();

  void compileAndRun(mlir::ModuleOp module, NonRestartingMLIRTimer &timer,
                     const llvm::DenseMap<mlir::Value, TensorId> &mappings);
  void addInput(const Buffer &ptr, const mlir::RankedTensorType &input,
                TensorId id);
  void readOutput(TensorId id, void *ptr);

  void freeTensor(TensorId id);

  ~PopitExecutor();

private:
  std::unique_ptr<PopitContext> _context;
};

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_
