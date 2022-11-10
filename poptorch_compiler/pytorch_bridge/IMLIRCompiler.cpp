// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "IMLIRCompiler.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

#include <deque>
#include <string>
#include <vector>

#include "passes/PassUtils.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"

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

bool isTrivialGraph(const mlir::ModuleOp &graph) {
  auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      const_cast<mlir::ModuleOp &>(graph).lookupSymbol(
          IMLIRCompiler::entry_point_name));
  ERROR_ON(!func);
  // If any of the blocks have an operation which isn't just an empty return
  // statement the graph may do something
  return llvm::all_of(func.getBody().getBlocks(), [](mlir::Block &block) {
    return llvm::all_of(block, [](mlir::Operation &op) {
      if (auto ret_op = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
        return ret_op.getNumOperands() == 0;
      }
      return false;
    });
  });
}

IMLIRCompiler::IMLIRCompiler(const poptorch::CompilerOptions &options)
    : root_timer(timing_manager.getRootTimer()),
      _builder(mlir::UnknownLoc::get(&context), &context),
      _compiler_options(&options) {

  context.getDiagEngine().registerHandler(printDiagnostic);

  // Load the dialects.
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<poptorch_ir::PoptorchDialect>();

  resetMainGraph();
}

bool IMLIRCompiler::isTrivialGraph() const {
  return poptorch_ir::detail::isTrivialGraph(_the_module);
}

void IMLIRCompiler::addGlobalState(std::string_view name,
                                   mlir::MemRefType argType) {
  _builder.setInsertionPointToEnd(&_the_module->getRegion(0).front());
  _builder.create<global_tensor_op>(
      /*sym_name=*/name,
      /*type=*/argType);
}

mlir::Value IMLIRCompiler::addArgument(mlir::func::FuncOp func,
                                       mlir::Type argType) {
  // Add the argument to the region.
  const auto insert_pos = func.getNumArguments();
  func.insertArgument(insert_pos, argType, {}, _builder.getLoc());

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
  case Type::NONE:
    return _builder.getNoneType();
  default:
    llvm::errs() << "Unreachable: Unsupported type.";
    exit(0);
  }
}

mlir::RankedTensorType
IMLIRCompiler::getTensor(Type type, const std::vector<std::int64_t> &dims) {
  return mlir::RankedTensorType::get(dims, convertType(type));
}

mlir::RankedTensorType IMLIRCompiler::getTensor(const TensorType &tensor_type) {
  return getTensor(tensor_type.element_type, tensor_type.shape);
}

TensorId IMLIRCompiler::addValue(const mlir::Value &value) {
  _value_map.push_back(value);
  return _value_map.size() - 1;
}

mlir::Value IMLIRCompiler::findValue(TensorId tensor) const {
  return _value_map.at(tensor);
}

void IMLIRCompiler::updateTensor(TensorId id, mlir::Value new_value) {
  _value_map[id] = new_value;
}

void IMLIRCompiler::resetMainGraph() {
  _the_module = mlir::ModuleOp::create(_builder.getLoc());
  _main_graph = createSubGraph(entry_point_name);

  // Invalidate all the values but do not clear the map:
  // the tensor IDs are still valid
  for (uint64_t i = 0; i < _value_map.size(); ++i) {
    _value_map[i] = mlir::Value();
  }
}

llvm::DenseMap<mlir::Value, TensorId> IMLIRCompiler::getValueMappings() const {
  llvm::DenseMap<mlir::Value, TensorId> mappings;
  for (std::uint64_t i = 0; i < _value_map.size(); ++i) {
    const auto val = _value_map[i];
    if (val) {
      ERROR_ON_MSG(!mappings.try_emplace(val, i).second,
                   "Value mapped to more than one TensorId");
    }
  }
  return mappings;
}

bool IMLIRCompiler::allOpsCanBeLoweredToPoplar() const {
  return _main_graph.all_ops_can_be_lowered;
}

IMLIRCompiler::Graph IMLIRCompiler::createSubGraph(std::string_view name) {
  auto func_type = _builder.getFunctionType({}, llvm::None);

  _builder.setInsertionPointToEnd(&_the_module->getRegion(0).front());
  Graph sub(_builder.create<mlir::func::FuncOp>(name, func_type));

  return sub;
}

} // namespace detail

} // namespace poptorch_ir
