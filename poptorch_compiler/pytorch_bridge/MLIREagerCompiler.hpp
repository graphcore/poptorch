// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_EAGER_COMPILER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_EAGER_COMPILER_HPP_

#include <vector>

#include "IMLIRCompiler.hpp"
#include "lower_to_poplar/PopitExecutor.hpp"

namespace poptorch_ir {

namespace detail {

class MLIREagerCompiler : public IMLIRCompiler {
public:
  explicit MLIREagerCompiler(const poptorch::CompilerOptions &options);

  virtual ~MLIREagerCompiler() = default;

  TensorId addInput(const Buffer &ptr, const mlir::RankedTensorType &input,
                    const char *name) override;

  TensorId addParameter(const Buffer &ptr,
                        const mlir::RankedTensorType &parameter,
                        const char *name) override;
  void addOutput(void *ptr, TensorId id, const char *name) override;
  void addReturn() override;
  void onOpAdded() override;
  TensorId addValue(const mlir::Value &value) override;
  mlir::Value findValue(TensorId tensor) override;
  void compileRunAndReset();

  bool shouldRunAllOpsSynchronously() const;

private:
  std::vector<mlir::RankedTensorType> _tensor_map;
  PopitExecutor _executor;
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_MLIR_EAGER_COMPILER_HPP_
