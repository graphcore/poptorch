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

  void initCompiler();
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

  at::Tensor detach(const at::Tensor &self) final;

  void registerEmptyTensor(const at::Tensor &tensor) final;

  const at::Tensor &copyInplace(const at::Tensor &self,
                                const at::Tensor &src) final;

  std::shared_ptr<MLIRExecutable> compile();

  poptorch_ir::TensorId findTensor(const at::Tensor &tensor);

  // Convert a stack of IValues into a vector of MLIR IR ids.
  std::vector<poptorch_ir::TensorId> mlirFromStack(c10::Stack &stack);

  // Some times pytorch specifies the output of an operation as an argument
  // without that operation being inplace, i.e matmul. In these cases we copy
  // and let the compiler eliminate it.
  at::Tensor outputIsInplaceOf(poptorch_ir::OptionalTensorId output_id,
                               const at::Tensor &original_input);

  // Some times pytorch specifies the output of an operation as an argument
  // without that operation being inplace, i.e matmul. In these cases we copy
  // and let the compiler eliminate it.
  std::vector<at::Tensor> outputIsInplaceOfList(
      const std::vector<poptorch_ir::OptionalTensorId> &output_id,
      const std::vector<at::Tensor> &original_input);

  // Handle the special case of squeeze_dim_, which is inplace in PyTorch but
  // changes the shape of the target tensor.
  // NOLINTNEXTLINE
  at::Tensor outputInplaceReshape_squeeze_dim_(poptorch_ir::TensorId output_id,
                                               const at::Tensor &original_input,
                                               poptorch_ir::TensorId self,
                                               int dim);

  at::Tensor makeEmptyOutputTensor(poptorch_ir::OptionalTensorId output_id,
                                   bool requires_grad);

  std::vector<at::Tensor> makeEmptyOutputTensorList(
      const std::vector<poptorch_ir::OptionalTensorId> &output_id,
      bool requires_grad);

// Add all the interface methods which match a single pytorch operation and
// convert it into MLIR.
#include "AtenToMlirInterface.hpp.inc"
#include "PoptorchToMlirInterface.hpp.inc"

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

  // We genenerate the lookup tables at object creation. This is the mechanism
  // by which that we use to target the right MLIR operation for a given aten
  // call.
  void generateDispatchTable();

  using ReturnTy = poptorch_ir::TensorId;

  // We have a set of handlers which just map an ATEN node directly onto an MLIR
  // operation.
  using StackFunctionType = std::function<void(c10::Stack &)>;
  std::unordered_map<std::string, StackFunctionType> _direct_dispatch_lookup;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_MLIR_DISPATCH_HPP_
