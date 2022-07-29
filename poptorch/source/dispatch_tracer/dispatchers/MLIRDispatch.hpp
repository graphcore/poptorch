// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_MLIR_DISPATCH_HPP_
#define POPTORCH_DISPATCH_MLIR_DISPATCH_HPP_
#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "poptorch/DispatchTracer.hpp"

// We don't build this on Centos TODO(T49566)
#if POPTORCH_BUILD_MLIR_COMPILER
#include "pytorch_bridge/PoptorchCompiler.hpp"
#endif

#include "../ValueMapper.hpp"
#include "IDispatch.hpp"

namespace poptorch {

class MLIRDispatch : public IDispatch {
public:
  MLIRDispatch();

  void initCompiler(bool eager_mode = false);
  at::Tensor addConstant(const at::Tensor &cpu_tensor) final;
  at::Tensor addInput(const at::Tensor &cpu_tensor) final;
  at::Tensor addParameter(const at::Tensor &cpu_tensor) final;
  void addOutput(const at::Tensor &ipu_src, const at::Tensor &cpu_dest) final;
  void createGraph() final;
  void finalizeGraph() final;

  void
  setCurrentCodeLocation(const torch::jit::SourceRange &source_location) final;
  void fallback(const c10::OperatorHandle &op, c10::Stack *stack) final;
  std::string handleOp(const c10::OperatorHandle &op, c10::Stack *stack);

  void detach(const c10::OperatorHandle &op, c10::Stack *stack,
              bool moving_parameters) final;

  void registerEmptyTensor(const at::Tensor &tensor) final;

  const at::Tensor &copyInplace(const at::Tensor &self,
                                const at::Tensor &src) final;

  std::shared_ptr<MLIRExecutor> compile();

  poptorch_ir::TensorId findTensor(const at::Tensor &tensor);

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

  // Handle the special case of ops which are inplace in PyTorch but change
  // the shape of the target tensor.
  // NOLINTNEXTLINE
  at::Tensor outputInplaceReshape(poptorch_ir::TensorId output_id,
                                  const at::Tensor &original_input,
                                  bool requires_grad);

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

  bool isEagerMode() { return _eager_mode; }

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

private:
// We don't build this on Centos TODO(T49566)
#if POPTORCH_BUILD_MLIR_COMPILER
  // The MLIR graph.
  poptorch_ir::PoptorchCompiler _compiler;
#endif

  // These are used to generate unique names
  // for each added input / output / parameter.
  uint64_t _next_input_idx{0};
  uint64_t _next_output_idx{0};
  uint64_t _next_parameter_idx{0};

  // We generate the lookup tables at object creation. This is the mechanism
  // by which we target the right MLIR operation for a given aten call.
  void generateDispatchTable();

  // We have a set of handlers which just map an ATEN node directly onto an MLIR
  // operation.
  using StackFunctionType = std::function<void(c10::Stack &)>;
  std::unordered_map<std::string, StackFunctionType> _direct_dispatch_lookup;
  bool _eager_mode{false};
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_MLIR_DISPATCH_HPP_
