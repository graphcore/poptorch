// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CompilerImpl.hpp"
#include <vector>

namespace poptorch_ir {
namespace detail {

PoptorchCompilerImpl::PoptorchCompilerImpl()
    : _builder(mlir::UnknownLoc::get(&context), &context),
      the_module(mlir::ModuleOp::create(_builder.getLoc())),
      executable(the_module) {

  // Load the dialect.
  context.loadDialect<poptorch_ir::PoptorchDialect>();

  // We represent our graph as a simple function.
  auto func_type = _builder.getFunctionType({}, llvm::None);
  main_graph = _builder.create<mlir::FuncOp>("MainGraph", func_type);
  the_module.push_back(main_graph);

  // Add an entry block.
  main_graph.addEntryBlock();

  // Same for write weights.
  write_weights_graph =
      _builder.create<mlir::FuncOp>("WeightsToDevice", func_type);
  main_graph.front().push_back(write_weights_graph);
  write_weights_graph.addEntryBlock();

  // Same for read weights.
  read_weights_graph =
      _builder.create<mlir::FuncOp>("WeightsToHost", func_type);
  main_graph.front().push_back(read_weights_graph);
  read_weights_graph.addEntryBlock();
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
  auto unsigned_ty = mlir::IntegerType::SignednessSemantics::Unsigned;
  auto signed_ty = mlir::IntegerType::SignednessSemantics::Signed;

  switch (type) {
  case Type::BOOL:
    return _builder.getIntegerType(1, signed_ty != 0u);
  case Type::CHAR:
  case Type::UNSIGNED_CHAR:
    return _builder.getIntegerType(8, signed_ty != 0u);
  case Type::SHORT:
    return _builder.getIntegerType(16, signed_ty != 0u);
  case Type::UNSIGNED_SHORT:
    return _builder.getIntegerType(16, unsigned_ty != 0u);
  case Type::UNSIGNED_INT:
    return _builder.getIntegerType(32, unsigned_ty != 0u);
  case Type::INT:
    return _builder.getIntegerType(32, signed_ty != 0u);
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
