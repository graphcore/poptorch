// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ADT/StringSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Value.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <string>
#include <utility>
#include <vector>

#include <poprithms/logging/timepartitionlogger.hpp>

// TODO(T49565): LLVM 13
// #include <mlir/Support/Timing.h>

#include "dialect/PoptorchDialect.hpp"
#include "lower_to_poplar/PoplarExecutor.hpp"
#include "pytorch_bridge/PoptorchCompiler.hpp"

namespace poptorch_ir {
namespace detail {

class PoptorchCompilerImpl {
public:
  PoptorchCompilerImpl();

  mlir::Type convertType(Type type);

  mlir::RankedTensorType getTensor(Type type,
                                   const std::vector<std::int64_t> &dims);

  // We have to jump through some hoops to add a new input after creation.
  // There's nicer ways of doing this in LLVM tree, once we upgrade should
  // change this.
  // TODO(T49565): Once we move from LLVM-13. See insertArgument in new API.
  mlir::Value addArgument(mlir::FuncOp func, mlir::Type argType);

  // We need to maintain some MLIR state.

  // The global context.
  mlir::MLIRContext context;

  // Builder to create ops.
  mlir::OpBuilder builder;

  // We don't have any code info so we just use unknown code location. Just a
  // helper so we don't have to recreate it every time.
  mlir::Location default_loc;

  // The main module which our functions are attached to.
  mlir::ModuleOp the_module;

  // The main graph.
  mlir::FuncOp main_graph;

  // The main block in the graph, we only have one.
  mlir::Block *main_graph_block;

  // Program to write weights onto the chip.
  mlir::FuncOp write_weights_graph;

  // Program to read weights off the chip.
  mlir::FuncOp read_weights_graph;

  // A mapping of SSA values to Poptorch IDs (the index in this vector)
  std::vector<mlir::Value> value_map;

  // Input and output callbacks to give to poplar.
  std::vector<std::string> input_callbacks;
  std::vector<std::pair<std::string, void *>> output_callbacks;
  std::vector<std::pair<std::string, void *>> weight_callbacks;

  // The executable.
  poptorch_ir::PoplarExecutable executable;

  poprithms::logging::ManualTimePartitionLogger timing_manager;

  // clang-format off
  // TODO(T49565): In LLVM 13 MLIR provides a really nice timing wrapper
  // which we can use and it integrates with all our passes.

  // A timer for us to record how long it takes to compile each stage.
  //mlir::DefaultTimingManager timing_manager_
  // Bit annoying, this shouldn't be needed really.
  // mlir::TimingScope root_timer;
  // A helper to provide a hidden interface to PopTorch to record how long it
  // takes to trace a model.
  // mlir::TimingScope tracer_timer;
  // clang-format on
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_
