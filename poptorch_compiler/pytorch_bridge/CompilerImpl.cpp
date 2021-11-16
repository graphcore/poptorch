// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CompilerImpl.hpp"
#include <vector>

namespace poptorch_ir {
namespace detail {

PoptorchCompilerImpl::PoptorchCompilerImpl()
    : builder(&context), default_loc(mlir::UnknownLoc::get(&context)),
      the_module(mlir::ModuleOp::create(default_loc)), executable(the_module) {
  // Load the dialect.
  context.loadDialect<poptorch_ir::PoptorchDialect>();

  // We represent our graph as a simple function.
  auto func_type = builder.getFunctionType({}, llvm::None);
  main_graph = mlir::FuncOp::create(default_loc, "MainGraph", func_type);
  the_module.push_back(main_graph);

  // Add an entry block.
  main_graph.addEntryBlock();

  // Same for write weights.
  write_weights_graph =
      mlir::FuncOp::create(default_loc, "WeightsToDevice", func_type);
  main_graph.front().push_back(write_weights_graph);
  write_weights_graph.addEntryBlock();

  // Same for read weights.
  read_weights_graph =
      mlir::FuncOp::create(default_loc, "WeightsToHost", func_type);
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
  mlir::FunctionType function_ty = builder.getFunctionType(types, llvm::None);

  // Give the function its new type.
  func.setType(function_ty);

  return val;
}

mlir::Type PoptorchCompilerImpl::convertType(Type type) {
  auto unsigned_ty = mlir::IntegerType::SignednessSemantics::Unsigned;
  auto signed_ty = mlir::IntegerType::SignednessSemantics::Signed;

  switch (type) {
  case Type::BOOL:
  case Type::CHAR:
  case Type::UNSIGNED_CHAR:
    return builder.getIntegerType(8, signed_ty != 0u);
  case Type::SHORT:
    return builder.getIntegerType(16, signed_ty != 0u);
  case Type::UNSIGNED_SHORT:
    return builder.getIntegerType(16, unsigned_ty != 0u);
  case Type::UNSIGNED_INT:
    return builder.getIntegerType(32, unsigned_ty != 0u);
  case Type::INT:
    return builder.getIntegerType(32, signed_ty != 0u);
  case Type::HALF:
    return builder.getF16Type();
  case Type::FLOAT:
    return builder.getF32Type();
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
