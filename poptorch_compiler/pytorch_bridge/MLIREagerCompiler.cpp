// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "MLIREagerCompiler.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LLVM.h>

#include <iterator>
#include <memory>

#include "dialect/Helpers.hpp"
#include "lower_to_poplar/EagerIpuSession.hpp"
#include "lower_to_poplar/IMLIRGraphConverter.hpp"
#include "lower_to_poplar/PopitExecutor.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/IpuSession.hpp"
#include "pytorch_bridge/PoptorchCompiler.hpp"

namespace poptorch_ir {

namespace detail {

namespace {

// Get the input and output tensor IDs from the graph. Note that this may fail
// if run after graph conversion and a pass in MLIRToPopitConverter replaces one
// of the graph inputs or outputs. We can't support this because we've got no
// way to map the new graph input to a Torch tensor.
auto extractInputsAndOutputs(
    mlir::ModuleOp module,
    const llvm::DenseMap<mlir::Value, TensorId> &mappings) {
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;

  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    inputs.clear();
    const auto argument_range = function.getArguments();
    inputs.reserve(argument_range.size());
    llvm::transform(argument_range, std::back_inserter(inputs),
                    [&](const mlir::Value &argument) {
                      auto it = mappings.find(argument);
                      ERROR_ON_MSG(
                          it == mappings.end(),
                          "[Internal] Input Value not found in tensor map");
                      return it->second;
                    });

    outputs.clear();
    function.walk([&](mlir::func::ReturnOp ret) {
      const auto result_range = ret.getOperands();
      outputs.reserve(result_range.size());
      llvm::transform(result_range, std::back_inserter(outputs),
                      [&](const mlir::Value &result) {
                        auto it = mappings.find(result);
                        ERROR_ON_MSG(
                            it == mappings.end(),
                            "[Internal] Output Value not found in tensor map");
                        return it->second;
                      });
    });
  }

  return std::pair(inputs, outputs);
}

} // namespace

MLIREagerCompiler::MLIREagerCompiler(const poptorch::CompilerOptions &options)
    : IMLIRCompiler(options) {}

TensorId MLIREagerCompiler::addValue(const mlir::Value &value) {
  ERROR_ON(!value);
  return IMLIRCompiler::addValue(value);
}

void MLIREagerCompiler::markOutputs(
    const llvm::DenseMap<mlir::Value, TensorId> &mappings,
    ILivenessMap &liveness) {
  std::vector<mlir::Value> outputs;

  // Handle overwriting of inputs by adding every input as an output. It will be
  // easier to filter these out later when the graph has value semantics.
  llvm::copy(_main_graph.graph.getArguments(), std::back_inserter(outputs));

  // Find all the outputs of all the ops as graph outputs.
  _main_graph.graph.walk([&](mlir::Operation *op) {
    // Don't add outputs for view tensors there will already be an output for
    // the viewed tensor
    if (op->hasTrait<::mlir::OpTrait::ViewOp>()) {
      return;
    }

    for (auto result : op->getResults()) {
      outputs.push_back(result);
    }
  });

  // Filter out erroneous graph outputs.
  outputs.erase(
      std::remove_if(
          outputs.begin(), outputs.end(),
          [&](const mlir::Value &output) {
            auto it = mappings.find(output);
            if (it == mappings.end()) {
              poptorch::logging::trace(
                  "No tensor ID mapping for {}: not marking as output",
                  mlirToStr(output));
              return true;
            }

            if (!liveness.extendLifetime(it->second)) {
              poptorch::logging::trace(
                  "Tensor {} is not alive in Python: not marking as output",
                  mlirToStr(output));
              return true;
            }

            return false;
          }),
      outputs.end());

  // Update the function's result types. These must match the arguments of the
  // return op added below.
  unsigned int result_idx = 0;
  for (const auto &output : outputs) {
    _main_graph.graph.insertResult(result_idx++, output.getType(), {});
  }

  createOp<mlir::func::ReturnOp>(outputs);
}

PopitDeviceFunctionWrapper MLIREagerCompiler::compile(EagerIpuSession &session,
                                                      ILivenessMap &liveness) {
  root_timer.start();

  auto mappings = getValueMappings();
  markOutputs(mappings, liveness);

  auto [input_ids, output_ids] = extractInputsAndOutputs(_the_module, mappings);

  // TODO(T69660): filter out unchanged outputs. This will be easier to do after
  // the passes getting rid of reference semantics have been applied. If every
  // op in the graph has value semantics any output that is also an input can
  // just be removed.

  // Run all the graph passes up to lowering.
  runGraphPasses(_the_module, root_timer);

  return session.func_cache.emplaceWrapped(_the_module, session, input_ids,
                                           output_ids, root_timer);
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

} // namespace detail
} // namespace poptorch_ir
