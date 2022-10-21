// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_
#define POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_

#include <llvm/ADT/DenseMap.h>

#include <memory>
#include <string>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/IpuSession.hpp"

#include "popit/popit.hpp"

namespace mlir {
class ModuleOp;
class TimingScope;
class Value;
class RankedTensorType;
} // namespace mlir

namespace poptorch_ir {

class EagerIpuSession;
class NonRestartingMLIRTimer;

class PopitDeviceFunction {
public:
  PopitDeviceFunction(EagerIpuSession &context, mlir::ModuleOp module,
                      NonRestartingMLIRTimer &timer,
                      const llvm::DenseMap<mlir::Value, TensorId> &mappings);

  void run(const std::vector<popit::Mem_t *> &inputs,
           const std::vector<popit::Mem_t *> &outputs);

  const std::vector<TensorId> &getInputs() const;
  const std::vector<TensorId> &getOutputs() const;

  friend class LowerToPopit;

private:
  // These attributes get populated by LowerToPopit
  popit::FunctionId_t _popit_fn;
  std::vector<TensorId> _input_ids;
  std::vector<TensorId> _output_ids;

  // Note we need to be careful that PopitFunctions aren't called after their
  // context is destroyed
  EagerIpuSession *_context;
};

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_POPIT_EXECUTOR_HPP_
