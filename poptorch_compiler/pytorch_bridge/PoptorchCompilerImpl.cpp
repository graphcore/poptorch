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
    : root_timer(timing_manager.getRootTimer()),
      _builder(mlir::UnknownLoc::get(&context), &context),
      _the_module(mlir::ModuleOp::create(_builder.getLoc())) {

  context.getDiagEngine().registerHandler(printDiagnostic);

  // Load the dialect.
  context.loadDialect<poptorch_ir::PoptorchDialect>();

  // We represent our graph as a simple function.
  auto func_type = _builder.getFunctionType({}, llvm::None);
  _main_graph.graph = _builder.create<mlir::FuncOp>("MainGraph", func_type);
  _the_module.push_back(_main_graph.graph);

  resetMainGraph();
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

TensorId PoptorchCompilerImpl::addValue(const mlir::Value &value) {
  _value_map.push_back(value);
  return _value_map.size() - 1;
}

mlir::Value PoptorchCompilerImpl::findValue(TensorId tensor) {
  return _value_map.at(tensor);
}

void PoptorchCompilerImpl::updateTensor(TensorId id, mlir::Value new_value) {
  _value_map[id] = new_value;
}

void PoptorchCompilerImpl::resetMainGraph() {
  // Clear the graph
  _main_graph.graph.eraseBody();
  // Add an entry block.
  _main_graph.graph.addEntryBlock();

  _main_graph.all_ops_can_be_lowered = true;

  // Invalidate all the values but do not clear the map:
  // the tensor IDs are still valid
  for (uint64_t i = 0; i < _value_map.size(); ++i) {
    _value_map[i] = mlir::Value();
  }
}

bool PoptorchCompilerImpl::allOpsCanBeLoweredToPoplar() const {
  return _main_graph.all_ops_can_be_lowered;
}

PoptorchCompilerImpl::Graph
PoptorchCompilerImpl::createSubGraph(const std::string &name) {
  Graph sub;
  auto func_type = _builder.getFunctionType({}, llvm::None);
  sub.graph = createOp<mlir::FuncOp>(name, func_type);
  sub.graph.addEntryBlock();
  return sub;
}

TensorId MLIREagerBuilder::addValue(const mlir::Value &value) {
  _tensor_map.push_back(value.getType().cast<mlir::RankedTensorType>());
  return PoptorchCompilerImpl::addValue(value);
}

mlir::Value MLIREagerBuilder::findValue(TensorId tensor) {
  mlir::Value value = PoptorchCompilerImpl::findValue(tensor);
  // This tensor comes from a previous graph, we need
  // to add it as an input to the new graph.
  if (!value) {
    value = addArgumentToMainGraph(_tensor_map.at(tensor));
    updateTensor(tensor, value);
  }
  return value;
}

void MLIREagerBuilder::compileRunAndReset() {
  // TODO(T57253) compile & run.
  resetMainGraph();
}

MLIRStaticGraphBuilder::MLIRStaticGraphBuilder() {
  _write_weights_graph = createSubGraph("WeightsToDevice");
  _read_weights_graph = createSubGraph("WeightsToHost");
}

// Compile graph by running both PopTorch compiler passes and poplar
// compilation.
poptorch_ir::PoplarExecutor
MLIRStaticGraphBuilder::compile(const PoplarTarget &target) {
  // Start the timer if it has not been started already
  timing_manager.setEnabled(true);
  root_timer.start();

  poptorch_ir::PoplarExecutor exe =
      compileExecutable(_the_module, target, root_timer);

  // End timing
  root_timer.stop();
  timing_manager.setEnabled(false);
  return exe;
}

void MLIRStaticGraphBuilder::addInput(const Buffer &ptr,
                                      const mlir::Value &input,
                                      const char *name) {
  // Add the argument to the function args.
  createOp<poptorch_ir::copy_from_host>(input, name);
  input_callbacks.push_back({name, ptr});
}

void MLIRStaticGraphBuilder::addParameter(const Buffer &ptr,
                                          const mlir::Value &parameter,
                                          const char *name) {
  // Write weights to the graph.
  createOp<poptorch_ir::copy_from_host>(_write_weights_graph, parameter,
                                        "Write-" + std::string(name));

  // Read weights from the graph.
  createOp<poptorch_ir::copy_to_host>(_read_weights_graph, parameter,
                                      "Read-" + std::string(name));

  weight_callbacks.push_back({name, ptr});
}
void MLIRStaticGraphBuilder::addOutput(void *ptr, const mlir::Value &output,
                                       const char *name) {
  createOp<poptorch_ir::copy_to_host>(output, name);
  output_callbacks.push_back({name, ptr});
}
void MLIRStaticGraphBuilder::addReturn() {
  createOp<poptorch_ir::end_graph>(_write_weights_graph);
  createOp<poptorch_ir::end_graph>(_read_weights_graph);
}
} // namespace detail

} // namespace poptorch_ir
