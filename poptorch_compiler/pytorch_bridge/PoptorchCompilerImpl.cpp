// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "PoptorchCompilerImpl.hpp"

#include <vector>

#include "passes/PassUtils.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {
namespace detail {

namespace {

mlir::LogicalResult printDiagnostic(mlir::Diagnostic &d) {
  poptorch::logging::Level lvl;
  switch (d.getSeverity()) {
  case mlir::DiagnosticSeverity::Error:
    lvl = poptorch::logging::Level::Err;
    break;
  case mlir::DiagnosticSeverity::Warning:
    lvl = poptorch::logging::Level::Warn;
    break;
  default:
    lvl = poptorch::logging::Level::Trace;
    break;
  }
  if (poptorch::logging::shouldLog(lvl)) {
    poptorch::logging::log(lvl, "{} [{}]", d.str(), mlirToStr(d.getLocation()));
  }
  return mlir::success();
}

} // namespace
mlir::Type getElementType(const mlir::Value &value) {
  return value.getType().cast<mlir::RankedTensorType>().getElementType();
}

bool higherThan(mlir::Type &lhs, mlir::Type &rhs) {
  ERROR_ON(!lhs);

  // Null always comes last
  if (!rhs) {
    return true;
  }

  // Both floats or both ints
  if ((lhs.isa<mlir::FloatType>() && rhs.isa<mlir::FloatType>()) ||
      (lhs.isa<mlir::IntegerType>() && rhs.isa<mlir::IntegerType>())) {
    return lhs.getIntOrFloatBitWidth() > rhs.getIntOrFloatBitWidth();
  }

  // Float always beats int
  if (lhs.isa<mlir::FloatType>() && rhs.isa<mlir::IntegerType>()) {
    return true;
  }

  if (lhs.isa<mlir::IntegerType>() && rhs.isa<mlir::FloatType>()) {
    return false;
  }

  ERROR("Unsupported types for implicit cast from " << mlirToStr(lhs) << " to "
                                                    << mlirToStr(rhs));
}

PoptorchCompilerImpl::PoptorchCompilerImpl()
    : _builder(mlir::UnknownLoc::get(&context), &context),
      _the_module(mlir::ModuleOp::create(_builder.getLoc())) {

  context.getDiagEngine().registerHandler(printDiagnostic);

  // Load the dialect.
  context.loadDialect<poptorch_ir::PoptorchDialect>();

  // We represent our graph as a simple function.
  auto func_type = _builder.getFunctionType({}, llvm::None);
  _main_graph = _builder.create<mlir::FuncOp>("MainGraph", func_type);
  _the_module.push_back(_main_graph);

  // Add an entry block.
  _main_graph.addEntryBlock();

  // Same for write weights.
  _write_weights_graph =
      _builder.create<mlir::FuncOp>("WeightsToDevice", func_type);
  _main_graph.front().push_back(_write_weights_graph);
  _write_weights_graph.addEntryBlock();

  // Same for read weights.
  _read_weights_graph =
      _builder.create<mlir::FuncOp>("WeightsToHost", func_type);
  _main_graph.front().push_back(_read_weights_graph);
  _read_weights_graph.addEntryBlock();
}

mlir::Value PoptorchCompilerImpl::addArgument(mlir::FuncOp func,
                                              mlir::Type argType) {
  // Add the argument to the region.
  const auto insert_pos = func.getNumArguments();
  func.insertArgument(insert_pos, argType, {});

  return func.getArgument(insert_pos);
}

mlir::Type PoptorchCompilerImpl::convertType(Type type) {
  switch (type) {
  case Type::BOOL:
    return _builder.getIntegerType(1, false);
  case Type::CHAR:
    return _builder.getIntegerType(8, true);
  case Type::UNSIGNED_CHAR:
    return _builder.getIntegerType(8, false);
  case Type::SHORT:
    return _builder.getIntegerType(16, true);
  case Type::UNSIGNED_SHORT:
    return _builder.getIntegerType(16, false);
  case Type::UNSIGNED_INT:
    return _builder.getIntegerType(32, false);
  case Type::INT:
    return _builder.getIntegerType(32, true);
  case Type::HALF:
    return _builder.getF16Type();
  case Type::FLOAT:
    return _builder.getF32Type();
  default:
    llvm::errs() << "Unreachable: Unsupported type.";
    exit(0);
  }
}

mlir::RankedTensorType
PoptorchCompilerImpl::getTensor(Type type,
                                const std::vector<std::int64_t> &dims) {
  return mlir::RankedTensorType::get(dims, convertType(type));
}

} // namespace detail

} // namespace poptorch_ir
