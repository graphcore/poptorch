// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CompilerImpl.hpp"
#include <vector>

namespace poptorch_ir {
namespace detail {

PoptorchCompilerImpl::PoptorchCompilerImpl()
    : _builder(mlir::UnknownLoc::get(&context), &context),
      _the_module(mlir::ModuleOp::create(_builder.getLoc())),
      _executable(_the_module) {

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
  mlir::Value val = func.getBody().addArgument(argType);

  // Rebuild the FunctionType from the above.
  llvm::SmallVector<mlir::Type> types;
  for (mlir::Value arg : func.front().getArguments()) {
    types.push_back(arg.getType());
  }

  // Create the new type.
  mlir::FunctionType function_ty = _builder.getFunctionType(types, llvm::None);

  // Give the function its new type.
  func.setType(function_ty);

  return val;
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
