// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "passes/LowerToPoplar.hpp"
#include "passes/PassUtils.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include <popops/Fill.hpp>
#include <poprand/RandomGen.hpp>

#include "dialect/PoptorchDialect.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {
/*
  Converts the MLIR graph into a poplar graph which can then be compiled.
 */
class LowerToPoplar final
    : public mlir::PassWrapper<LowerToPoplar,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  LowerToPoplar() {}

  LowerToPoplar(poplar::Graph &g, CompilerContext &c)
      : _graph(&g), _context(&c) {
    poplin::addCodelets(g);
    popnn::addCodelets(g);
    popops::addCodelets(g);
    poprand::addCodelets(g);
  }

  void runOnOperation() override;

  poplar::Graph &graph() { return *_graph; }

private:
  // Verify operations using MLIR's verifier.
  static void verifyOperations(const mlir::FuncOp &function);

  poplar::Graph *_graph;

  poptorch_ir::CompilerContext *_context;
};

// Poplar keeps the size and shape as two distinct concepts.
struct PoplarTypePair {
  poplar::Type element_type;
  std::vector<std::size_t> shape;
};

// Mlir type to poplar type helper.
poplar::Type elementTypeFromMLIR(mlir::Type elementType) {
  if (elementType.isF16()) {
    return poplar::HALF;
  }
  if (elementType.isF32()) {
    return poplar::FLOAT;
  }
  if (elementType.isUnsignedInteger(8)) {
    return poplar::UNSIGNED_CHAR;
  }
  if (elementType.isUnsignedInteger(16)) {
    return poplar::UNSIGNED_SHORT;
  }
  if (elementType.isUnsignedInteger(32) || elementType.isUnsignedInteger(64)) {
    return poplar::UNSIGNED_INT;
  }
  // We use isInteger from here onwards to capture both
  // isSignedInteger and isSignlessInteger
  if (elementType.isInteger(1)) {
    return poplar::BOOL;
  }
  if (elementType.isInteger(8)) {
    return poplar::SIGNED_CHAR;
  }
  if (elementType.isInteger(16)) {
    return poplar::SHORT;
  }
  if (elementType.isInteger(32) || elementType.isInteger(64)) {
    return poplar::INT;
  }
  assert(false && "Unsupported MLIR type");

  return poplar::FLOAT;
}

PoplarTypePair processType(mlir::Type mlirType) {
  // Turn it into a ranked tensor.
  mlir::RankedTensorType tensor_type = mlirType.cast<mlir::RankedTensorType>();
  mlir::Type element_type = tensor_type.getElementType();

  // Extract the element type of the tensor.
  poplar::Type type = elementTypeFromMLIR(element_type);

  // Convert the dimensions into something poplar understands.
  std::vector<std::size_t> dims;
  for (int64_t dim : tensor_type.getShape()) {
    dims.push_back(dim);
  }

  return {type, dims};
}

void LowerToPoplar::runOnOperation() {
  mlir::ModuleOp module = this->getOperation();

  poptorch::logging::info("Graph lowered to poplar:\n{}", mlirOpToStr(module));

  for (mlir::FuncOp function : module.getOps<mlir::FuncOp>()) {
    verifyOperations(function);
    _context->seq = poplar::program::Sequence();

    // Walk over all functions with a poplar impl.
    function.walk(
        [&](PoplarImplInterface impl) { impl.lowerToPoplar(*_context); });

    _context->programs[Programs::MainGraph] = _context->seq;

    // For the read/write subfunctions do the same.
    for (mlir::FuncOp subfunc : function.getOps<mlir::FuncOp>()) {
      Programs program = poptorch_ir::Programs::WeightsToDevice;
      if (subfunc.getName() == "WeightsToDevice") {
        program = poptorch_ir::Programs::WeightsToDevice;
      } else if (subfunc.getName() == "WeightsToHost") {
        program = poptorch_ir::Programs::WeightsToHost;
      }

      _context->seq = poplar::program::Sequence();

      // Walk over all functions with a poplar impl.
      subfunc.walk(
          [&](PoplarImplInterface impl) { impl.lowerToPoplar(*_context); });

      _context->programs[program] = _context->seq;
    }
  }
}

void LowerToPoplar::verifyOperations(const mlir::FuncOp &function) {
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

// Get the poplar tensor which corresponds to a specific value of MLIR.
poplar::Tensor CompilerContext::fromSsa(mlir::Value value) {
  auto itr = tensors.find(value);
  if (itr != tensors.end()) {
    return itr->second;
  }

  const PoplarTypePair tensor_type = processType(value.getType());

  // Actually add the tensor to the graph.
  poplar::Tensor tensor =
      this->graph.addVariable(tensor_type.element_type, tensor_type.shape,
                              poplar::VariableMappingMethod::LINEAR);

  tensors.insert({value, tensor});
  return tensor;
}

std::vector<poplar::Tensor>
CompilerContext::fromSsa(mlir::ValueRange value_range) {
  std::vector<poplar::Tensor> poplar_tensors;

  for (mlir::Value value : value_range) {
    poplar_tensors.push_back(fromSsa(value));
  }

  return poplar_tensors;
}

poplar::Tensor &CompilerContext::getRandomSeed() {
  // NOTE: This mechanism is a temporary workaround while TODO(T51096) remains
  //       unresolved, to handle loading, saving & restoring of the seed.
  if (!_randomSeed) {
    _randomSeed = graph.addVariable(poplar::UNSIGNED_INT, {2},
                                    poplar::VariableMappingMethod::LINEAR);
    popops::fill(graph, *_randomSeed, seq, 42);
  }

  return *_randomSeed;
}

poplar::Type CompilerContext::poplarTypeOf(mlir::Type elementType) {
  return elementTypeFromMLIR(elementType);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerToPoplarPass(poplar::Graph &graph, CompilerContext &context) {
  return std::make_unique<LowerToPoplar>(graph, context);
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::LowerToPoplar>
    lower("lower-to-poplar", "");
