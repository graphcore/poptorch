// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/PopitExecutor.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
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
#include <popit/popit.hpp>

#include "CompilerHelpers.hpp"
#include "lower_to_poplar/EagerIpuSession.hpp"
#include "lower_to_poplar/IMLIRGraphConverter.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "passes/LowerToPopit.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "poptorch_logging/LoggingLight.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/IpuSession.hpp"

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

} // namespace

PopitDeviceFunction::PopitDeviceFunction(EagerIpuSession &context,
                                         mlir::ModuleOp module,
                                         NonRestartingMLIRTimer &timer)
    : _context(&context) {

  auto compile_popit = timer.nestAndScope("Compiling popit");

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

} // namespace poptorch_ir
