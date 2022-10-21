// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_EAGER_COMPILER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_EAGER_COMPILER_HPP_

#include <vector>

#include "IMLIRCompiler.hpp"
#include "pytorch_bridge/IpuSession.hpp"

namespace poptorch_ir {

class EagerIpuSession;
class ILivenessMap;

namespace detail {

class MLIREagerCompiler : public IMLIRCompiler {
public:
  explicit MLIREagerCompiler(const poptorch::CompilerOptions &options);

  TensorId addInput(const mlir::RankedTensorType &input,
                    const char *name) override;

  TensorId addParameter(Buffer &ptr, const mlir::RankedTensorType &parameter,
                        const char *name) override;
  void addOutput(TensorId id, const char *name) override;
  void addReturn() override;
  TensorId addValue(const mlir::Value &value) override;

  PopitDeviceFunctionWrapper compile(EagerIpuSession &session,
                                     const ILivenessMap &liveness);

private:
  void markOutputs(const llvm::DenseMap<mlir::Value, TensorId> &mappings,
                   const ILivenessMap &liveness);
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_EAGER_COMPILER_HPP_
