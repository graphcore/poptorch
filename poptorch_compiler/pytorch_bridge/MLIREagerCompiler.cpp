// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "MLIREagerCompiler.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <memory>

#include "lower_to_poplar/PopitExecutor.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/PoptorchCompiler.hpp"

namespace poptorch_ir {
namespace detail {

MLIREagerCompiler::MLIREagerCompiler(const poptorch::CompilerOptions &options)
    : IMLIRCompiler(options) {}

TensorId MLIREagerCompiler::addValue(const mlir::Value &value) {
  ERROR_ON(!value);
  return IMLIRCompiler::addValue(value);
}

void MLIREagerCompiler::markOutputs(
    const llvm::DenseMap<mlir::Value, TensorId> &mappings,
    const ILivenessMap &liveness) {
  std::vector<mlir::Value> output_ids;

  // Handle overwriting of inputs
  for (const auto &argument : _main_graph.graph.getArguments()) {
    const auto is_overwritten = [&](const mlir::OpOperand &operand) {
      auto overwrite_op = mlir::dyn_cast_or_null<overwrite>(operand.getOwner());
      return overwrite_op && argument == overwrite_op.dest();
    };
    if (llvm::any_of(argument.getUses(), is_overwritten)) {
      output_ids.push_back(argument);
    }
  }

  // Find all the outputs of all the ops as graph outputs
  _main_graph.graph.walk([&](mlir::Operation *op) {
    for (auto result : op->getResults()) {
      output_ids.push_back(result);
    }
  });

  // Add them all as graph outputs
  for (auto &output : output_ids) {
    auto it = mappings.find(output);
    if (it == mappings.end()) {
      poptorch::logging::trace(
          "No tensor ID mapping for {}: not marking as output",
          mlirToStr(output));
    } else if (!liveness.isAlive(it->second)) {
      poptorch::logging::trace(
          "Tensor {} is not alive in python: not marking as output",
          mlirToStr(output));
    } else {
      createOp<poptorch_ir::output_tensor>(output, it->second);
    }
  }
}

PopitDeviceFunctionWrapper
MLIREagerCompiler::compile(EagerIpuSession &session,
                           const ILivenessMap &liveness) {
  root_timer.start();

  auto mappings = getValueMappings();

  markOutputs(mappings, liveness);

  return PopitDeviceFunctionWrapper(std::make_unique<PopitDeviceFunction>(
      session, _the_module, root_timer, mappings));
}

TensorId MLIREagerCompiler::addInput(const mlir::RankedTensorType &input,
                                     const char *name) {
  UNUSED(name);

  auto value = addArgumentToMainGraph(input);
  const TensorId id = IMLIRCompiler::addValue(value);

  return id;
}

TensorId MLIREagerCompiler::addParameter(
    Buffer &ptr, const mlir::RankedTensorType &parameter, const char *name) {
  UNUSED(ptr);
  return addInput(parameter, name);
}

void MLIREagerCompiler::addOutput(TensorId id, const char *name) {
  UNUSED(id);
  UNUSED(name);
}

void MLIREagerCompiler::addReturn() {
  ERROR("Only static graphs have a return");
}
} // namespace detail

} // namespace poptorch_ir
