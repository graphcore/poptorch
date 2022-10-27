// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/IpuSession.hpp"
#include "pytorch_bridge/PoplarExecutorWrapper.hpp"

namespace mlir {
class RankedTensorType;
class Operation;
} // namespace mlir

namespace poptorch {
struct CompilerOptions;
class ValueMapper;
} // namespace poptorch

namespace poptorch_ir {

class Buffer;

namespace detail {
class IMLIRCompiler;
} // namespace detail

enum class ExecutionType { StaticGraph, EagerMode };

enum class CompilerBackend { Poplar, PopIR };

class ILivenessMap {
public:
  virtual ~ILivenessMap() = default;

  // Check if the tensor is currently alive and extend it's lifetime if it is.
  // We need to extend the lifetime to ensure that the tensor doesn't get
  // destroyed between it being marked as an output and the graph being ran
  virtual bool extendLifetime(TensorId id) = 0;
};

class PoptorchCompiler {
public:
  PoptorchCompiler();
  ~PoptorchCompiler();

  PoptorchCompiler(PoptorchCompiler &&other);
  PoptorchCompiler(const PoptorchCompiler &other) = delete;
  PoptorchCompiler &operator=(PoptorchCompiler &&other);
  PoptorchCompiler &operator=(const PoptorchCompiler &other) = delete;

  void startTraceTiming();
  void endTraceTiming();
  void getTimingInfo();

  void dump();

  void init(ExecutionType execution_type, CompilerBackend compiler_backend,
            const poptorch::CompilerOptions &options);

  void setCurrentPythonCodeLocation(const char *filename, std::uint64_t line,
                                    std::uint64_t col);

  TensorId addInput(const TensorType &type, const char *);
  TensorId addParameter(Buffer &ptr, const TensorType &type, const char *);
  void addOutput(TensorId id, const char *);

  // Only if ExecutionType::EagerMode is used.
  bool isTrivialGraph() const;
  // Only if ExecutionType::EagerMode is used.
  PopitDeviceFunctionWrapper compile(IIpuSession &session,
                                     ILivenessMap &liveness);

  // Only if ExecutionType::StaticGraph is used
  std::unique_ptr<PoplarExecutorWrapper> compileAndLoad();

  std::vector<std::int64_t> getSize(TensorId id) const;
  Type getType(TensorId id) const;

  // Non popart.
  void addReturn();

  bool isView(TensorId id) const;

  // Return true if all the ops in the graph can be lowered to
  // Poplar.
  bool allOpsCanBeLoweredToPoplar() const;

// Each tablegen entry will automatically generate a C++ method and impl which
// can be used by PyTorch. This means Compiler will have a function to add any
// op using non-pytorch, non-mlir types. Tensors are poptorch_ir::TensorId.
// Functions return void, poptorch_ir::TensorId, or
// std::vector<poptorch_ir::TensorId> depending on their type.
#include "dialect/AutogenCompiler.hpp.inc"

private:
  mlir::RankedTensorType getRankedTensorType(TensorId id) const;

  std::unique_ptr<detail::IMLIRCompiler> _impl;
};

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_HPP_
