// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/PopitExecutor.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <thread>
#include <utility>

#include <passes/CommonPasses.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>

#include <popit/Device.hpp>

#include "CompilerHelpers.hpp"
#include "lower_to_poplar/IMLIRGraphConverter.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "lower_to_poplar/PopitSession.hpp"
#include "passes/LowerToPopit.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch_ir {

namespace {
class MLIRToPopitConverter final : public IMLIRGraphConverter {
public:
  explicit MLIRToPopitConverter(PopitDeviceFunction &func) : _func(func) {}

protected:
  void addCustomPasses(mlir::PassManager &manager) override {
    manager.addPass(createLowerToPopitPass(_func));
  }

private:
  PopitDeviceFunction &_func;
};

auto extractInputsAndOutputs(
    mlir::ModuleOp module,
    const llvm::DenseMap<mlir::Value, TensorId> &mappings) {
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;

  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    inputs.clear();
    outputs.clear();
    inputs.reserve(function.getArguments().size());
    llvm::transform(function.getArguments(), std::back_inserter(inputs),
                    [&](const mlir::Value &argument) {
                      auto it = mappings.find(argument);
                      // This can only happen if a pass in MLIRToPopitConverter
                      // replaces one of the graph inputs. We can't support this
                      // because we've got no way to map the new graph input to
                      // a Torch tensor.
                      ERROR_ON_MSG(
                          it == mappings.end(),
                          "[Internal] Input Value not found in tensor map");
                      return it->second;
                    });
    function.walk([&](poptorch_ir::output_tensor output) {
      const auto it = mappings.find(output.tensor());
      // This can only happen if a pass in MLIRToPopitConverter
      // replaces one of the graph inputs. We can't support this
      // because we've got no way to map the new graph input to
      // a Torch tensor.
      ERROR_ON_MSG(it == mappings.end(),
                   "[Internal] Output Value not found in tensor map");

      outputs.emplace_back(it->second);
    });
  }

  return std::pair(inputs, outputs);
}
} // namespace

PopitDeviceFunction::PopitDeviceFunction(
    EagerIpuSession &context, mlir::ModuleOp module,
    NonRestartingMLIRTimer &timer,
    const llvm::DenseMap<mlir::Value, TensorId> &mappings)
    : _context(&context) {

  auto compile_popit = timer.nestAndScope("Compiling popit");

  std::tie(_input_ids, _output_ids) = extractInputsAndOutputs(module, mappings);

  MLIRToPopitConverter converter{*this};
  converter.convertGraph(module, timer);
  compile_popit.stop();
}

void PopitDeviceFunction::run(const std::vector<popit::Mem_t *> &inputs,
                              const std::vector<popit::Mem_t *> &outputs) {
  // Execute the function
  popit::call(_context->session.get(), _popit_fn,
              /* ipuIndex=*/0, inputs, outputs);
}

const std::vector<TensorId> &PopitDeviceFunction::getOutputs() const {
  return _output_ids;
}
const std::vector<TensorId> &PopitDeviceFunction::getInputs() const {
  return _input_ids;
}
} // namespace poptorch_ir
