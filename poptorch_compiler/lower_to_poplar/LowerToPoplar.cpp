// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "lower_to_poplar/LowerToPoplar.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

namespace {
/*
  Converts the MLIR graph into a poplar graph which can then be compiled.
 */
class LowerToPoplar
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

  void addCopyForInput(mlir::Value value);

  poplar::Graph &graph() { return *_graph; }

private:
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
  if (elementType.isInteger(8)) {
    return poplar::SIGNED_CHAR;
  }
  if (elementType.isSignedInteger(8)) {
    return poplar::CHAR;
  }
  if (elementType.isSignedInteger(16)) {
    return poplar::SHORT;
  }
  if (elementType.isSignedInteger(32)) {
    return poplar::INT;
  }
  if (elementType.isSignedInteger(64)) {
    return poplar::LONG;
  }
  if (elementType.isUnsignedInteger(8)) {
    return poplar::UNSIGNED_CHAR;
  }
  if (elementType.isUnsignedInteger(16)) {
    return poplar::UNSIGNED_SHORT;
  }
  if (elementType.isUnsignedInteger(32)) {
    return poplar::UNSIGNED_INT;
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
  mlir::ModuleOp the_module = this->getOperation();

  the_module->dump();

  for (mlir::FuncOp function : the_module.getOps<mlir::FuncOp>()) {
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerToPoplarPass(poplar::Graph &graph, CompilerContext &context) {
  return std::make_unique<LowerToPoplar>(graph, context);
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::LowerToPoplar> lower("", "");
