// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "IMLIRCompiler.hpp"

#include <deque>
#include <string>
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

IMLIRCompiler::IMLIRCompiler()
    : root_timer(timing_manager.getRootTimer()),
      _builder(mlir::UnknownLoc::get(&context), &context) {

  context.getDiagEngine().registerHandler(printDiagnostic);

  // Load the dialect.
  context.loadDialect<poptorch_ir::PoptorchDialect>();

  resetMainGraph();
}

mlir::Value IMLIRCompiler::addArgument(mlir::FuncOp func, mlir::Type argType) {
  // Add the argument to the region.
  const auto insert_pos = func.getNumArguments();
  func.insertArgument(insert_pos, argType, {});

  return func.getArgument(insert_pos);
}

mlir::Value IMLIRCompiler::addArgumentToMainGraph(mlir::Type argType) {
  return addArgument(_main_graph.graph, argType);
}

mlir::Type IMLIRCompiler::convertType(Type type) {
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
IMLIRCompiler::getTensor(Type type, const std::vector<std::int64_t> &dims) {
  return mlir::RankedTensorType::get(dims, convertType(type));
}

TensorId IMLIRCompiler::addValue(const mlir::Value &value) {
  _value_map.push_back(value);
  return _value_map.size() - 1;
}

mlir::Value IMLIRCompiler::findValue(TensorId tensor) {
  return _value_map.at(tensor);
}

void IMLIRCompiler::updateTensor(TensorId id, mlir::Value new_value) {
  _value_map[id] = new_value;
}

void IMLIRCompiler::resetMainGraph() {
  _the_module = mlir::ModuleOp::create(_builder.getLoc());
  // We represent our graph as a simple function.
  auto func_type = _builder.getFunctionType({}, llvm::None);
  _main_graph.graph = _builder.create<mlir::FuncOp>("MainGraph", func_type);
  _the_module.push_back(_main_graph.graph);

  // Add an entry block.
  _main_graph.graph.addEntryBlock();
  _main_graph.all_ops_can_be_lowered = true;

  // Invalidate all the values but do not clear the map:
  // the tensor IDs are still valid
  for (uint64_t i = 0; i < _value_map.size(); ++i) {
    _value_map[i] = mlir::Value();
  }
}

llvm::DenseMap<mlir::Value, TensorId> IMLIRCompiler::getValueMappings() {
  llvm::DenseMap<mlir::Value, TensorId> mappings;
  for (std::uint64_t i = 0; i < _value_map.size(); ++i) {
    if (_value_map.at(i)) {
      ERROR_ON_MSG(!mappings.try_emplace(_value_map.at(i), i).second,
                   "Value mapped to more than one TensorId");
    }
  }
  return mappings;
}

bool IMLIRCompiler::allOpsCanBeLoweredToPoplar() const {
  return _main_graph.all_ops_can_be_lowered;
}

IMLIRCompiler::Graph IMLIRCompiler::createSubGraph(const std::string &name) {
  Graph sub;
  auto func_type = _builder.getFunctionType({}, llvm::None);
  sub.graph = createOp<mlir::FuncOp>(name, func_type);
  sub.graph.addEntryBlock();
  return sub;
}

} // namespace detail

} // namespace poptorch_ir
