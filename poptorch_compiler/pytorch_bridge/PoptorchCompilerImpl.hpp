// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/Timing.h>

#include <llvm/ADT/StringSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Value.h>

#include <mlir/IR/BuiltinOps.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dialect/PoptorchDialect.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "lower_to_poplar/PopitExecutor.hpp"
#include "lower_to_poplar/PoplarExecutor.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/PoptorchCompiler.hpp"

namespace poptorch_ir {

namespace detail {

// Returns the element type of an MLIR value.
mlir::Type getElementType(const mlir::Value &value);

// Converts an MLIR type to a string representation.
std::string mlirTypeToStr(mlir::Type &type);

// Whether lhs is a higher ranked type than lhs: if true, rhs should be
// implicitly casted to lhs (or a higher ranked type than both).
bool higherThan(mlir::Type &lhs, mlir::Type &rhs);

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

// Implementation of a for loop for the next function
// Equivalent code for start = 0
// void fn(OpTy, f, args) {
//   for(size_t idx = 0; idx < end; idx++) {
//     if(implicitCatsOn<OpTy, idx>) {
//       f(args[start].getType().cast<mlir:RankedTensorType>().getElementType());
//     }
//   }
// }
template <typename OpTy, size_t start, size_t end, typename F, typename... Args>
void callOnEachElementType(F &&f, const Args &...args) {
  if constexpr (start < end) {
    if constexpr (implicitCastsOn<OpTy, start>()) {
      f(std::get<start>(std::forward_as_tuple(args...))
            .getType()
            .template cast<mlir::RankedTensorType>()
            .getElementType());
    }
    callOnEachElementType<OpTy, start + 1, end>(f, args...);
  }
}

// Returns the type to which the compiler should promote all implicit casting
// operands
template <typename OpTy, typename... Args>
mlir::Type getCastToType(const mlir::MLIRContext & /*new_context*/,
                         const Args &...args) {
  // Maintain the highest type
  mlir::Type highest_type;

  // Loop on the number of operands, or the maximum number which can be
  // implicitly cast, whichever is higher
  constexpr size_t num_loops =
      std::max(sizeof...(Args), mlir::OpTrait::max_implicit_casting_operands);

  // Loop over all operands and update highest_type if necessary.
  callOnEachElementType<OpTy, 0, num_loops>(
      [&highest_type](mlir::Type new_type) {
        if (higherThan(new_type, highest_type)) {
          highest_type = new_type;
        }
      },
      args...);

  ERROR_ON(!highest_type);
  return highest_type;
}

// Implicit cast an mlir::Value to promote_to if necessary, otherwise simply
// forward element.
template <typename OpTy, std::size_t Ind, typename Elt>
auto castAndForwardElt(mlir::ImplicitLocOpBuilder &builder,
                       mlir::FuncOp &main_graph, const mlir::Type &promote_to,
                       Elt &&element) {
  if constexpr (implicitCastsOn<OpTy, Ind>()) {
    static_assert(
        std::is_same<typename std::remove_reference<decltype(element)>::type,
                     mlir::Value>::value,
        "Only an mlir::Value can be implicitly casted.");

    mlir::Type element_type = element.getType()
                                  .template cast<mlir::RankedTensorType>()
                                  .getElementType();

    if (element_type != promote_to) {
      // Create a casting op for the implicit casting and add it to the graph
      // updating the operand to the promoted output.
      // (poptorch_ir::cast is an automatically generated mlir::Op subclass.)
      auto casted = builder.create<poptorch_ir::cast>(element, promote_to);
      main_graph.front().push_back(casted);
      return casted.result();
    }
  }

  // Use perfect forwarding in other cases.
  return std::forward<Elt>(element);
}

// Perform the looping for implicit casting by unfolding over indices.
// Equivalent code
// ArgsTuple castAndForward(OpTy, builder, main_graph, promote_to, old_args) {
//   ArgsTuple new_args;
//   for(auto idx: old_args.size()) {
//     if(implicitCatsOn<OpTy, idx>) {
//        if(old_args[idx].type() != promote_to) {
//          auto casted = builder.create<cast>(old_args[idx], promote_to);
//          main_graph.front().push_back(tmp);
//          new_args.append(casted.result());
//          continue;
//        }
//     }
//     new_args.append(old_arg[idx]);
//   }
//   return new_args;
// }
template <typename OpTy, typename... Args, std::size_t... Inds>
auto castAndForwardImpl(mlir::ImplicitLocOpBuilder &builder,
                        mlir::FuncOp &main_graph, const mlir::Type &promote_to,
                        std::tuple<Args...> &&input_tuple,
                        std::index_sequence<Inds...> /*unused*/) {
  // Calls castAndForwardElt on every argument through unwidning, with "Inds"
  // becoming the given index, and returns a tuple of the lot.
  return std::make_tuple(castAndForwardElt<OpTy, Inds>(
      builder, main_graph, promote_to, std::get<Inds>(input_tuple))...);
}

// Loop through all operands, promote to promote_to if required for implicit
// casting, and then return all operands as a tuple.
// This uses template recursion, performing the promotions on the inital call
// and then returning the final value.
template <typename OpTy, typename... Args>
auto castAndForward(mlir::ImplicitLocOpBuilder &builder,
                    mlir::FuncOp &main_graph, const mlir::Type &promote_to,
                    std::tuple<Args...> &&input_tuple) {
  return castAndForwardImpl<OpTy>(builder, main_graph, promote_to,
                                  std::move(input_tuple),
                                  std::make_index_sequence<sizeof...(Args)>());
}

// Implicitly cast all implicit casting operands to promote_to and then create a
// new op of class OpTy using the (possibly promoted) operands.
// Uses the builder and main_graph to add the implicitly casting ops.
// Consistent with other methods, this does *not* add the resulting op to
// main_graph, though all the implicit casting ops are.
template <typename OpTy, typename... Args>
OpTy implicitCastAndCreate(mlir::ImplicitLocOpBuilder &builder,
                           mlir::FuncOp &main_graph,
                           const mlir::Type &promote_to, Args &&...args) {
  // Call the builder create method by unpacking the tuple into arguments
  // The tuple is formed all the (possibly promoted) operands, args,
  // from calling castAndForward.
  auto call_builder = [&builder](Args &&...args_inner) {
    return builder.create<OpTy>(args_inner...);
  };
  return std::apply(call_builder,
                    castAndForward<OpTy>(builder, main_graph, promote_to,
                                         std::forward_as_tuple(args...)));
}

// TODO(T57253): rename to IMLIRGraphBuilder?
class PoptorchCompilerImpl {
public:
  PoptorchCompilerImpl();
  virtual ~PoptorchCompilerImpl() = default;

  mlir::Type convertType(Type type);

  mlir::RankedTensorType getTensor(Type type,
                                   const std::vector<std::int64_t> &dims);

  static mlir::Value addArgument(mlir::FuncOp func, mlir::Type argType);

  mlir::Value addArgumentToMainGraph(mlir::Type argType);

  virtual TensorId addInput(const Buffer &ptr,
                            const mlir::RankedTensorType &input,
                            const char *name) = 0;

  virtual TensorId addParameter(const Buffer &ptr,
                                const mlir::RankedTensorType &parameter,
                                const char *name) = 0;
  virtual void addOutput(void *ptr, TensorId id, const char *name) = 0;
  virtual void addReturn() = 0;

  virtual void onOpAdded() {}
  // Print module to stderr
  void dump() { _the_module->dump(); }

  // Set the source code location (file line and col)
  // The MLIR ImplicitLocOpBuilder maintains a source code location so that
  // the location does not be sent as part of creating an op. This method allows
  // the location to be set.
  void setLoc(const char *filename, std::uint64_t line, std::uint64_t col) {
    _builder.setLoc(mlir::FileLineColLoc::get(
        _builder.getContext(), _builder.getIdentifier(filename), line, col));
  }

  template <typename OpTy, typename... Args> OpTy createOp(Args &&...args) {
    return createOp<OpTy>(_main_graph, std::forward<Args>(args)...);
  }

  virtual TensorId addValue(const mlir::Value &value);

  // Can't be const because in some cases it might trigger some graph
  // modification.
  virtual mlir::Value findValue(TensorId tensor);

  bool allOpsCanBeLoweredToPoplar() const;

protected:
  struct Graph {
    mlir::FuncOp graph;
    // When a new op is added to the main graph using createOp we check and
    // store whether or not there is an actual handler for this op. (Some ops
    // will have been added with only shape inference and no implementation, in
    // which case we won't be able to lower them later on).
    bool all_ops_can_be_lowered{true};
  };

  // Update the MLIR value associated to a tensor id.
  // This is typically needed when a tensor was created
  // by a graph and then used by another one: the tensor
  // would be an op output in the first graph and a graph
  // input in the second.
  void updateTensor(TensorId id, mlir::Value new_value);
  // Remove all the ops in the main graph but do not reset
  // the tensor IDs. Any valid tensor ID passed to findValue()
  // will now return an empty Value.
  void resetMainGraph();

  Graph createSubGraph(const std::string &name);

  // Create a new op of class OpTy, possibly casting operands specified by args.
  template <typename OpTy, typename... Args>
  OpTy createOp(Graph &graph, Args &&...args) {
    if constexpr (isImplicitCastingOp<OpTy>()) {
      mlir::Type promote_to = getCastToType<OpTy>(context, args...);
      OpTy op = implicitCastAndCreate<OpTy>(_builder, graph.graph, promote_to,
                                            std::forward<Args>(args)...);
      graph.graph.front().push_back(op);
      return op;
    }

    OpTy op = _builder.create<OpTy>(std::forward<Args>(args)...);

    graph.all_ops_can_be_lowered &=
        !OpTy::template hasTrait<mlir::OpTrait::NotImplementedOp>();
    graph.graph.front().push_back(op);
    return op;
  }

  llvm::DenseMap<mlir::Value, TensorId> getValueMappings();

public:
  // We need to maintain some MLIR state.
  // The global context.
  mlir::MLIRContext context;

  // A timer for us to record how long it takes to compile each stage.
  mlir::DefaultTimingManager timing_manager;

  // Wrapped root timer, which does not restart if start is called twice.
  NonRestartingMLIRTimer root_timer;

  // A helper to provide a hidden interface to PopTorch to record how long it
  // takes to trace a model.
  mlir::TimingScope tracer_timer;

  // Empty the main graph, delete all the values but does not reset the tensor
  // IDs.
private:
  // A mapping of SSA values to Poptorch IDs (the index in this vector)
  std::vector<mlir::Value> _value_map;
  // Builder to create ops.
  mlir::ImplicitLocOpBuilder _builder;

protected:
  // The main graph.
  Graph _main_graph;

  // The main module which our functions are attached to.
  mlir::ModuleOp _the_module;
};

class MLIRStaticGraphBuilder : public PoptorchCompilerImpl {
public:
  MLIRStaticGraphBuilder();
  virtual ~MLIRStaticGraphBuilder() = default;
  // Compile graph by running both PopTorch compiler passes and poplar
  // compilation.
  poptorch_ir::PoplarExecutor compile(const PoplarTarget &target);
  TensorId addInput(const Buffer &ptr, const mlir::RankedTensorType &input,
                    const char *name) override;

  TensorId addParameter(const Buffer &ptr,
                        const mlir::RankedTensorType &parameter,
                        const char *name) override;
  void addOutput(void *ptr, TensorId id, const char *name) override;
  void addReturn() override;

private:
  // Program to write weights onto the chip.
  Graph _write_weights_graph;

  // Program to read weights off the chip.
  Graph _read_weights_graph;

public:
  // Input and output callbacks to give to poplar.
  std::vector<std::pair<std::string, Buffer>> input_callbacks;
  std::vector<std::pair<std::string, void *>> output_callbacks;
  std::vector<std::pair<std::string, Buffer>> weight_callbacks;
};

class MLIREagerBuilder : public PoptorchCompilerImpl {
public:
  explicit MLIREagerBuilder(PoplarDevice &device);
  virtual ~MLIREagerBuilder() = default;

  TensorId addInput(const Buffer &ptr, const mlir::RankedTensorType &input,
                    const char *name) override;

  TensorId addParameter(const Buffer &ptr,
                        const mlir::RankedTensorType &parameter,
                        const char *name) override;
  void addOutput(void *ptr, TensorId id, const char *name) override;
  void addReturn() override;
  void onOpAdded() override;
  TensorId addValue(const mlir::Value &value) override;
  mlir::Value findValue(TensorId tensor) override;
  void compileRunAndReset();

private:
  std::vector<mlir::RankedTensorType> _tensor_map;
  PopitExecutor _executor;
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_IMPL_HPP_
