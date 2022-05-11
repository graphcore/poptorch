// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ADT/StringSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
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
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/PoptorchCompiler.hpp"

namespace poptorch_ir {
enum class AddToGraph { MAIN_GRAPH = 0, READ_WEIGHTS, WRITE_WEIGHTS };

namespace detail {

// Returns whether the op with class OpTy will implicitly cast the operand on
// the specified index, idx.
template <typename OpTy, size_t idx> constexpr bool implicitCastsOn() {
  return OpTy::template hasTrait<
      mlir::OpTrait::ImplicitCastOperand<idx>::template Impl>();
}

// Implementation of a for-each loop for the next function
template <typename OpTy, std::size_t... Idxs>
/// Equivalent code
//  bool isImplicitCastingOpForLoop(OpTy, Idxs) {
//   for(auto idx : Idxs) {
//     if(implicitCatsOn<OpTy, idx>) { return true; }
//   }
//   return false;
// }
constexpr bool
isImplicitCastingOpForEachLoop(std::index_sequence<Idxs...> /*unused*/) {
  // Unfold all the indices into a long "or" expression.
  // NB the actual parameter is "unused" but the type is used as the "Idx..."
  // represents a compile time sequence of indices.
  return (... || implicitCastsOn<OpTy, Idxs>());
}

// Returns whether OpTy is an implicing casting op: an op which requires
// operands specified as implicit casting to be promoted, if necessary, to a
// common type.
template <typename OpTy> constexpr bool isImplicitCastingOp() {
  return isImplicitCastingOpForEachLoop<OpTy>(
      std::make_index_sequence<mlir::OpTrait::max_implicit_casting_operands>());
}

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

  mlir::Value addArgumentToMainGraph(mlir::Type argType) {
    return addArgument(_main_graph, argType);
  }

  // Print module to stderr
  void dump() { _the_module->dump(); }

  // Set the source code location (file line and col)
  // The MLIR ImplicitLocOpBuilder maintains a source code location so that
  // the location does not be sent as part of creating an op. This method allows
  // the location to be set.
  void setLoc(const char *filename, std::uint64_t line, std::uint64_t col) {
    _builder.setLoc(_builder.getFileLineColLoc(_builder.getIdentifier(filename),
                                               line, col));
  }

  // Create a new op
  template <typename OpTy, typename... Args>
  OpTy createOp(AddToGraph add_to_graph, Args &&...args) {
    if (isImplicitCastingOp<OpTy>()) {
      poptorch::logging::warn(
          "Implicit casting not yet implemented, affected op: {}",
          OpTy::getOperationName().str());
    }

    OpTy op = _builder.create<OpTy>(std::forward<Args>(args)...);

    switch (add_to_graph) {
    case AddToGraph::MAIN_GRAPH:
      all_ops_can_be_lowered &=
          !OpTy::template hasTrait<mlir::OpTrait::NotImplementedOp>();
      _main_graph.front().push_back(op);
      break;
    case AddToGraph::READ_WEIGHTS:
      _read_weights_graph.front().push_back(op);
      break;
    case AddToGraph::WRITE_WEIGHTS:
      _write_weights_graph.front().push_back(op);
      break;
    default:
      ERROR("Invalid value for add_to_graph");
    }
    return op;
  }

  // Compile graph by running both PopTorch compiler passes and poplar
  // compilation.
  void compile() {
    ERROR_ON(!_executable);
    _executable.compile(timing_manager);
  }

  // Connect to a poplar stream with a fixed location in memory.
  // Each time Poplar copies data to/from the named stream, it will read/write
  // to/from this memory locaiton.
  void connectStream(const std::string &string, void *ptr) {
    ERROR_ON(!_executable);
    _executable.connectStream(string, ptr);
  }

  void connectStream(const std::string &string, Buffer buffer) {
    ERROR_ON(!_executable);
    _executable.connectStream(string, std::move(buffer));
  }

  // Return the executable such that it can be moved out this class
  poptorch_ir::PoplarExecutable &&getExecutable() {
    return std::move(_executable);
  }

  // We need to maintain some MLIR state.

  // The global context.
  mlir::MLIRContext context;

  // A mapping of SSA values to Poptorch IDs (the index in this vector)
  std::vector<mlir::Value> value_map;

  // Input and output callbacks to give to poplar.
  std::vector<std::pair<std::string, Buffer>> input_callbacks;
  std::vector<std::pair<std::string, void *>> output_callbacks;
  std::vector<std::pair<std::string, Buffer>> weight_callbacks;

  poprithms::logging::ManualTimePartitionLogger timing_manager;
  // When a new op is added to the main graph using createOp we check and
  // store whether or not there is an actual handler for this op. (Some ops will
  // have been added with only shape inference and no implementation, in which
  // case we won't be able to lower them later on).
  bool all_ops_can_be_lowered{true};

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

private:
  // Builder to create ops.
  mlir::ImplicitLocOpBuilder _builder;

  // The main module which our functions are attached to.
  mlir::ModuleOp _the_module;

  // The main graph.
  mlir::FuncOp _main_graph;

  // Program to write weights onto the chip.
  mlir::FuncOp _write_weights_graph;

  // Program to read weights off the chip.
  mlir::FuncOp _read_weights_graph;

  // The executable.
  poptorch_ir::PoplarExecutable _executable;
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_
