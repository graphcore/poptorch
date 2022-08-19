// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "LowerToPopit.hpp"

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

#include "../CompilerHelpers.hpp"
#include "../PopitContext.hpp"
#include "passes/PassUtils.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Fill.hpp>
#include <poprand/RandomGen.hpp>

#include "dialect/PoptorchDialect.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {

/*
  Converts the MLIR graph into a poplar graph which can then be compiled
  and executed by PopIT.
 */
class LowerToPopit final
    : public mlir::PassWrapper<LowerToPopit,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void init(PopitContext *context) { _context = context; }

  void runOnOperation() override;

private:
  // Verify operations using MLIR's verifier.
  static void verifyOperations(const mlir::FuncOp &function);

  PopitContext *_context;
};

void LowerToPopit::runOnOperation() {
  mlir::ModuleOp module = this->getOperation();

  poptorch::logging::info("Graph lowered to popit:\n{}", mlirOpToStr(module));

  bool first_function = true;
  for (mlir::FuncOp function : module.getOps<mlir::FuncOp>()) {
    ERROR_ON_MSG(!first_function, "More than one function in the module");
    first_function = false;
    verifyOperations(function);

    // Find all the inputs
    _context->inputs.clear();
    _context->inputs.reserve(function.getNumArguments());
    for (auto &argument : function.getArguments()) {
      _context->inputs.push_back(argument);
    }

    // Create the PopIT specs.
    std::vector<popit::TensorSpec> input_specs;
    input_specs.reserve(_context->inputs.size());
    for (auto &input : _context->inputs) {
      input_specs.push_back(getTensorSpec(input.getType()));
    }

    // This lambda will actually be called by popitCall, so we need to make
    // sure it doesn't depend on "this" as this object will not exist anymore
    // and it will segfault.
    PopitContext *ctx = _context;
    auto popit_fn =
        [&](poplar::Graph &graph, std::vector<poplar::Tensor> &tensors,
            poplar::program::Sequence &program) -> std::vector<poplar::Tensor> {
      CompilerContext context(graph, program);

      // Add all the graph inputs to the context.
      ERROR_ON_MSG(ctx->inputs.size() != tensors.size(),
                   "Number of inputs mismatch");
      for (std::uint64_t i = 0; i < tensors.size(); i++) {
        context.addTensor(ctx->inputs.at(i), tensors.at(i));
      }

      // Walk over all functions with a poplar impl.
      function.walk(
          [&](PoplarImplInterface impl) { impl.lowerToPoplar(context); });

      // All the tensors that were written to are outputs.
      std::vector<poplar::Tensor> outputs;
      ctx->output_ids.clear();
      function.walk([&](poptorch_ir::output_tensor output) {
        outputs.push_back(context.fromSsa(output.tensor()));
        ctx->output_ids.push_back(output.tensorIdAttr().getInt());
      });

      return outputs;
    };

    // Add the function
    _context->popit_fn =
        popit::addFunction(_context->session.get(), input_specs, /*inouts=*/{},
                           popit_fn, function.getName().str());
  }
}

void LowerToPopit::verifyOperations(const mlir::FuncOp &function) {
  // It would be possible to call mlir::verify on the whole graph, however
  // this would not pinpoint the failing operation. Therefore, we verify each
  // op at a time, by recursing into it. Note that calling mlir::verify has some
  // extra checks ommited here.
  auto num_regions = function->getNumRegions();
  for (unsigned i = 0; i < num_regions; i++) {
    auto &region = function->getRegion(i);

    for (auto &block : region) {
      for (auto &op : block) {
        std::string op_name = op.getName().getStringRef().str();

        // WeightsToDevice/Host are FuncOps, which means that they should be
        // isolated from external variables. However, this is not the case so
        // verification would fail. There is no interface to remove a trait
        // so they would have to be defined as some other type such as a
        // custom defined non-isolated function op in the tablegen. Instead,
        // we simply skip verification. The name will be "func" when calling
        // getName on the mlir::Operation (here) rather than the mlir::funcOp.
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

} // namespace

popit::TensorSpec getTensorSpec(mlir::Type mlirType) {
  // Turn it into a ranked tensor.
  mlir::RankedTensorType tensor_type = mlirType.cast<mlir::RankedTensorType>();
  mlir::Type element_type = tensor_type.getElementType();

  // Extract the element type of the tensor.
  poplar::Type type = elementTypeFromMLIR(element_type);

  // Convert the dimensions into something poplar understands.
  popit::TensorShape shape;
  for (int64_t dim : tensor_type.getShape()) {
    shape.push_back(static_cast<std::uint64_t>(dim));
  }

  return {type, shape};
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerToPopitPass(PopitContext &context) {
  auto pass = std::make_unique<LowerToPopit>();
  pass->init(&context);
  return std::move(pass);
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::LowerToPopit> lower("lower-to-popit",
                                                               "");
