// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "LowerToPopit.hpp"

#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"

#include "../CompilerHelpers.hpp"
#include "passes/PassUtils.hpp"

#include <popit/functions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Fill.hpp>
#include <poprand/RandomGen.hpp>

#include "dialect/PoptorchDialect.hpp"
#include "lower_to_poplar/EagerIpuSession.hpp"
#include "lower_to_poplar/PopitExecutor.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

/*
  Converts the MLIR graph into a poplar graph which can then be compiled
  and executed by PopIT.
 */
class LowerToPopit final
    : public mlir::PassWrapper<LowerToPopit,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit LowerToPopit(PopitDeviceFunction *func) : _func(func) {}

  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "lower-to-popit"; }

  mlir::StringRef getDescription() const override {
    return "Construct a PopIT graph from the given MLIR graph. Does not modify "
           "the input graph.";
  }

private:
  // Verify operations using MLIR's verifier.
  static void verifyOperations(const mlir::func::FuncOp &function);

  PopitDeviceFunction *_func;
};

void LowerToPopit::runOnOperation() {
  mlir::ModuleOp module = this->getOperation();

  poptorch::logging::info("Graph lowered to popit:\n{}", mlirOpToStr(module));

  bool first_function = true;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    ERROR_ON_MSG(!first_function, "More than one function in the module");
    first_function = false;
    verifyOperations(function);

    // Find all the inputs and create the PopIT specs.
    std::vector<popit::TensorSpec> input_specs;
    input_specs.reserve(function.getArguments().size());
    llvm::transform(
        function.getArguments(), std::back_inserter(input_specs),
        [](const auto &argument) { return getTensorSpec(argument.getType()); });

    auto popit_fn =
        [&function](
            poplar::Graph &graph, std::vector<poplar::Tensor> &tensors,
            poplar::program::Sequence &program) -> std::vector<poplar::Tensor> {
      CompilerContext context(graph, program);

      // Add all the graph inputs to the context.
      ERROR_ON_MSG(function.getArguments().size() != tensors.size(),
                   "Number of inputs mismatch");
      for (const auto &[input, poplar_tensor] :
           llvm::zip(function.getArguments(), tensors)) {
        context.addTensor(input, poplar_tensor);
      }

      // Walk over all functions with a poplar impl.
      function.walk(
          [&](PoplarImplInterface impl) { impl.lowerToPoplar(context); });

      // All the tensors that were written to are outputs.
      std::vector<poplar::Tensor> outputs;
      function.walk([&](poptorch_ir::output_tensor output) {
        outputs.push_back(context.fromSsa(output.tensor()));
      });

      return outputs;
    };

    // Add the function
    _func->_popit_fn =
        popit::addFunction(_func->_context->session.get(), input_specs,
                           /*inouts=*/{}, popit_fn, function.getName().str());
  }
}

void LowerToPopit::verifyOperations(const mlir::func::FuncOp &function) {
  // It would be possible to call mlir::verify on the whole graph, however
  // this would not pinpoint the failing operation. Therefore, we verify each
  // op at a time, by recursing into it. Note that calling mlir::verify has some
  // extra checks ommited here.
  auto num_regions = function->getNumRegions();
  for (unsigned i = 0; i < num_regions; i++) {
    auto &region = function->getRegion(i);

    for (auto &block : region) {
      for (auto &op : block) {
        const std::string op_name = op.getName().getStringRef().str();

        // WeightsToDevice/Host are FuncOps, which means that they should be
        // isolated from external variables. However, this is not the case so
        // verification would fail. There is no interface to remove a trait
        // so they would have to be defined as some other type such as a
        // custom defined non-isolated function op in the tablegen. Instead,
        // we simply skip verification. The name will be "func" when calling
        // getName on the mlir::Operation (here) rather than the
        // mlir::func::FuncOp.
        if (op_name == "func") {
          continue;
        }

        if (mlir::failed(mlir::verify(&op))) {
          ERROR("Verification failed for " << op_name << ":\n "
                                           << mlirOpToStr(op));
        }
      }
    }
  }
}

popit::TensorSpec getTensorSpec(mlir::Type mlirType) {
  // Turn it into a ranked tensor.
  mlir::RankedTensorType const tensor_type =
      mlirType.cast<mlir::RankedTensorType>();
  mlir::Type const element_type = tensor_type.getElementType();

  // Extract the element type of the tensor.
  poplar::Type const type = elementTypeFromMLIR(element_type);

  // Convert the dimensions into something poplar understands.
  popit::TensorShape shape;
  for (int64_t dim : tensor_type.getShape()) {
    shape.push_back(static_cast<std::uint64_t>(dim));
  }

  return {type, shape};
}

// Note: this pass isn't registered because it doesn't have any side effect
// visible in the graph
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerToPopitPass(PopitDeviceFunction &func) {
  auto pass = std::make_unique<LowerToPopit>(&func);
  return std::move(pass);
}

} // namespace poptorch_ir
