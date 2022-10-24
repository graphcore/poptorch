// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_MLIR_DISPATCH_HPP_
#define POPTORCH_DISPATCH_MLIR_DISPATCH_HPP_
#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "poptorch/DispatchTracer.hpp"
#include "pytorch_bridge/CompilerOptions.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/PoptorchCompiler.hpp"

#include "../ValueMapper.hpp"
#include "IDispatch.hpp"

namespace poptorch {

class MLIRDispatch : public IDispatch {
public:
  MLIRDispatch(const CompilerOptions &options, TensorStore *tensor_store);

  void addConstant(const at::Tensor &cpu_tensor,
                   const at::Tensor &ipu_tensor) final;
  void addInput(const at::Tensor &cpu_tensor,
                const at::Tensor &ipu_tensor) final;
  void addParameter(const at::Tensor &cpu_tensor,
                    const at::Tensor &ipu_tensor) final;
  void addOutput(const at::Tensor &ipu_src, const at::Tensor &cpu_dest) final;
  void finalizeGraph() final;
  void markStep() final;

  void fallback(const c10::OperatorHandle &op, c10::Stack *stack) final;
  std::string handleOp(const c10::OperatorHandle &op, c10::Stack *stack);

  void detach(const c10::OperatorHandle &op, c10::Stack *stack,
              bool moving_parameters) final;

  void promoteAsParameter(const at::Tensor &tensor);

  void promoteAsInput(const at::Tensor &tensor, bool is_wrapped = false);

  void promoteAsOutput(const at::Tensor &tensor);

  poptorch_ir::TensorId addEmptyTensorOp(const at::Tensor &tensor);

  void registerEmptyTensor(const at::Tensor &tensor) final;

  std::shared_ptr<MLIRExecutor> compile();

  poptorch_ir::TensorId findTensor(const at::Tensor &tensor);
  std::vector<poptorch_ir::TensorId>
  findTensor(const std::vector<at::Tensor> &tensors);

  // Some times pytorch specifies the output of an operation as an argument
  // without that operation being inplace, i.e matmul. In these cases we copy
  // and let the compiler eliminate it.
  at::Tensor outputIsInplaceOf(poptorch_ir::OptionalTensorId output_id,
                               const at::Tensor &original_input,
                               bool requires_grad);

  // Some times pytorch specifies the output of an operation as an argument
  // without that operation being inplace, i.e matmul. In these cases we copy
  // and let the compiler eliminate it.
  std::vector<at::Tensor> outputIsInplaceOfList(
      const std::vector<poptorch_ir::OptionalTensorId> &output_id,
      const std::vector<at::Tensor> &original_input,
      const std::vector<bool> &requires_grad);

  // Compute whether grad is required based on a list of requires_grad
  // determination types.  The argument requires_grad_or should be true if any
  // of the input tensors to the op had requires_grad=true (this will be used
  // if requires_grad_types[i] == OR_INPUTS).
  static std::vector<bool> requiresGrad(
      const std::vector<poptorch_ir::RequiresGradType> &requires_grad_types,
      bool requires_grad_or);

  at::Tensor makeEmptyOutputTensor(poptorch_ir::OptionalTensorId output_id,
                                   bool requires_grad);

  std::vector<at::Tensor> makeEmptyOutputTensorList(
      const std::vector<poptorch_ir::OptionalTensorId> &output_id,
      const std::vector<bool> &requires_grad);

  bool isEagerMode() const;
  bool shouldRunAllOpsSynchronously() const;
  bool extractOutputImmediately() const;
  CompilerOptions &getMutableCompilerOptions();
  const std::vector<std::vector<char>> &getSourceLocationExcludes() const final;

// Add all the interface methods which match a single pytorch operation and
// convert it into MLIR.
#include "AtenToMLIRInterface.hpp.inc"
#include "PoptorchToMLIRInterface.hpp.inc"
#include "TorchScatterToMLIRInterface.hpp.inc"

protected:
  // Returns the only tensor ID in the vector, if it exists, or
  // poptorch::OptionalTensorId others. Errors if the vector is longer than
  // length one.
  static poptorch_ir::TensorId getSingleOptionalTensorId(
      const std::vector<poptorch_ir::OptionalTensorId> &tensor_vec);

  // Find any tensors which were created before this Dispatch was created, and
  // promote them by marking them as inputs to the graph.
  void findAndPromoteExternalTensors(c10::Stack *stack);

private:
  void
  setCurrentCodeLocation(const torch::jit::SourceRange &source_location) final;

  // Reset the dispatcher back to first construction
  void reset();

  void initCompiler(const CompilerOptions &compilerOptions);

  // The MLIR graph.
  poptorch_ir::PoptorchCompiler _compiler;

  // These are used to generate unique names
  // for each added input / output / parameter.
  uint64_t _next_input_idx{0};
  uint64_t _next_output_idx{0};
  uint64_t _next_parameter_idx{0};

public:
  using StackFunctionType = std::function<void(MLIRDispatch &, c10::Stack &)>;
  using DispatchTable = std::unordered_map<std::string, StackFunctionType>;

private:
  // We have a set of handlers which just map an ATEN node directly onto an MLIR
  // operation.
  static const DispatchTable direct_dispatch_lookup;

  CompilerOptions _opts;

  std::vector<IpuTensorDetails *> _aliases_to_restore;
  void restoreAliases();

  bool isDeferredEmptyTensor(const at::Tensor &tensor) const;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_MLIR_DISPATCH_HPP_
