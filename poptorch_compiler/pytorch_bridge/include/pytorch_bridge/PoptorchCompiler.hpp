// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/Executor.hpp"

namespace mlir {
class RankedTensorType;
}

namespace poptorch_ir {

namespace detail {
class PoptorchCompilerImpl;
}

class PoptorchCompiler {
public:
  PoptorchCompiler();
  ~PoptorchCompiler();

  void startTraceTiming();
  void endTraceTiming();
  void getTimingInfo();

  void dump();

  void init();

  TensorId addInput(void *ptr, const std::vector<std::int64_t> &shape, Type,
                    const char *);
  TensorId addParameter(void *ptr, const std::vector<std::int64_t> &shape, Type,
                        const char *);
  TensorId addBuffer(void *ptr, const std::vector<std::int64_t> &shape, Type,
                     const char *);

  void addOutput(TensorId id, void *ptr, const char *);

  PoptorchExecutorWrapper compile();

  std::vector<std::int64_t> getSize(TensorId id) const;
  Type getType(TensorId id) const;

  // Non popart.
  void addReturn();

  bool isView(TensorId id) const;

// Each tablegen entry will automatically generate a C++ method and impl which
// can be used by PyTorch. This means Compiler will have a function to add any
// op using non-pytorch, non-mlir types. Tensors are poptorch_ir::TensorId.
// Functions return void, poptorch_ir::TensorId, or
// std::vector<poptorch_ir::TensorId> depending on their type.
#include "dialect/AutogenCompiler.hpp.inc"

private:
  mlir::RankedTensorType getRankedTensorType(TensorId id) const;

  std::unique_ptr<detail::PoptorchCompilerImpl> _impl;
};

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_
