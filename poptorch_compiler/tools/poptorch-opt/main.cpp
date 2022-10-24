// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include "dialect/PoptorchDialect.hpp"

int main(int argc, char **argv) {
  mlir::registerCanonicalizerPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, poptorch_ir::PoptorchDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "poptorch-opt: poptorch MLIR optimiser", registry));
}
