// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_IMLIR_COMPILER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_IMLIR_COMPILER_HPP_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/Timing.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dialect/PoptorchDialect.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "pytorch_bridge/CompilerOptions.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch_ir {

class Buffer;

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
template <typename T, typename F> struct CallWithElementTypeImpl {
  static void call(const T &t, F &&f) {
    if (t.getImpl() == nullptr) {
      return;
    }
    f(t.getType().template cast<mlir::RankedTensorType>().getElementType());
  }
};

template <typename F>
struct CallWithElementTypeImpl<llvm::SmallVector<mlir::Value, 4>, F> {
  static void call(const llvm::SmallVector<mlir::Value, 4> &t, F &&f) {
    for (size_t i = 0; i < t.size(); i++) {
      CallWithElementTypeImpl<mlir::Value, F>::call(t[i], std::forward<F>(f));
    }
  }
};

template <typename T, typename F>
using CallWithElementType =
    CallWithElementTypeImpl<std::remove_const_t<std::remove_reference_t<T>>, F>;

template <typename OpTy, size_t start, size_t end, typename F, typename... Args>
void callOnEachElementType(F &&f, const Args &...args) {
  if constexpr (start < end) {
    if constexpr (implicitCastsOn<OpTy, start>()) {
      auto tup = std::forward_as_tuple(args...);
      CallWithElementType<std::tuple_element_t<start, decltype(tup)>, F>::call(
          std::get<start>(tup), std::forward<F>(f));
    }
    callOnEachElementType<OpTy, start + 1, end>(f, args...);
  }
}

// Returns the type to which the compiler should promote all implicit casting
// operands
template <typename OpTy, typename... Args>
mlir::Type getCastToType(mlir::MLIRContext &context, const Args &...args) {
  if (OpTy::template hasTrait<mlir::OpTrait::ImplicitCastToBool>()) {
    return mlir::IntegerType::get(&context, 1, mlir::IntegerType::Unsigned);
  }
  // Maintain the highest type
  mlir::Type highest_type;

  if (OpTy::template hasTrait<mlir::OpTrait::ImplicitCastToFloat>()) {
    highest_type = mlir::FloatType::getF32(&context);
  }

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
struct CastAndForwardElt {
  static auto f(mlir::ImplicitLocOpBuilder &builder,
                const mlir::Type &promote_to, Elt &&element) {
    if constexpr (implicitCastsOn<OpTy, Ind>()) {
      static_assert(
          std::is_same<typename std::remove_reference<decltype(element)>::type,
                       mlir::Value>::value,
          "Only an mlir::Value can be implicitly casted.");
      if (element.getImpl() == nullptr) {
        return std::forward<Elt>(element);
      }
      mlir::Type element_type = element.getType()
                                    .template cast<mlir::RankedTensorType>()
                                    .getElementType();

      if (element_type != promote_to) {
        // Create a casting op for the implicit casting and add it to the graph
        // updating the operand to the promoted output.
        // (poptorch_ir::cast is an automatically generated mlir::Op subclass.)
        auto casted = builder.create<poptorch_ir::cast>(element, promote_to);
        return casted.result();
      }
    }

    // Use perfect forwarding in other cases.
    return std::forward<Elt>(element);
  }
};

template <typename OpTy, std::size_t Ind>
struct CastAndForwardElt<OpTy, Ind, llvm::SmallVector<mlir::Value, 4> &> {
  static auto f(mlir::ImplicitLocOpBuilder &builder,
                const mlir::Type &promote_to,
                llvm::SmallVector<mlir::Value, 4> element) {
    if constexpr (implicitCastsOn<OpTy, Ind>()) {
      llvm::SmallVector<mlir::Value, 4> result;
      for (size_t i = 0; i < element.size(); i++) {
        result.emplace_back(CastAndForwardElt<OpTy, Ind, mlir::Value &>::f(
            builder, promote_to, element[i]));
      }
      return result;
    }
    return element;
  }
};

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
                        const mlir::Type &promote_to,
                        std::tuple<Args...> &&input_tuple,
                        std::index_sequence<Inds...> /*unused*/) {
  // Calls castAndForwardElt on every argument through unwidning, with "Inds"
  // becoming the given index, and returns a tuple of the lot.
  return std::make_tuple(
      CastAndForwardElt<OpTy, Inds,
                        std::tuple_element_t<Inds, std::tuple<Args...>>>::
          f(builder, promote_to, std::get<Inds>(input_tuple))...);
}

// Loop through all operands, promote to promote_to if required for implicit
// casting, and then return all operands as a tuple.
// This uses template recursion, performing the promotions on the inital call
// and then returning the final value.
template <typename OpTy, typename... Args>
auto castAndForward(mlir::ImplicitLocOpBuilder &builder,
                    const mlir::Type &promote_to,
                    std::tuple<Args...> &&input_tuple) {
  return castAndForwardImpl<OpTy>(builder, promote_to, std::move(input_tuple),
                                  std::make_index_sequence<sizeof...(Args)>());
}

// Implicitly cast all implicit casting operands to promote_to and then create a
// new op of class OpTy using the (possibly promoted) operands.
// Uses the builder and main_graph to add the implicitly casting ops.
// Consistent with other methods, this does *not* add the resulting op to
// main_graph, though all the implicit casting ops are.
template <typename OpTy, typename... Args>
OpTy implicitCastAndCreate(mlir::ImplicitLocOpBuilder &builder,
                           const mlir::Type &promote_to, Args &&...args) {
  // Call the builder create method by unpacking the tuple into arguments
  // The tuple is formed all the (possibly promoted) operands, args,
  // from calling castAndForward.
  auto call_builder = [&builder](Args &&...args_inner) {
    return builder.create<OpTy>(args_inner...);
  };
  return std::apply(call_builder,
                    castAndForward<OpTy>(builder, promote_to,
                                         std::forward_as_tuple(args...)));
}

class IMLIRCompiler {
public:
  explicit IMLIRCompiler(const poptorch::CompilerOptions &options);
  virtual ~IMLIRCompiler() = default;

  bool isTrivialGraph() const;

  mlir::Type convertType(Type type);

  mlir::RankedTensorType getTensor(Type type,
                                   const std::vector<std::int64_t> &dims);
  mlir::RankedTensorType getTensor(const TensorType &tensor_type);

  mlir::Value addArgument(mlir::func::FuncOp func, mlir::Type argType);

  mlir::Value addArgumentToMainGraph(mlir::Type argType);

  virtual TensorId addInput(const mlir::RankedTensorType &input,
                            const char *name) = 0;

  virtual TensorId addParameter(Buffer &ptr,
                                const mlir::RankedTensorType &parameter,
                                const char *name) = 0;
  virtual void addOutput(TensorId id, const char *name) = 0;
  virtual void addReturn() = 0;

  // Print module to stderr
  void dump() { _the_module->dump(); }

  // Set the source code location (file line and col)
  // The MLIR ImplicitLocOpBuilder maintains a source code location so that
  // the location does not be sent as part of creating an op. This method allows
  // the location to be set.
  void setLoc(const char *filename, std::uint64_t line, std::uint64_t col) {
    if (filename != nullptr) {
      _builder.setLoc(mlir::FileLineColLoc::get(_builder.getContext(), filename,
                                                line, col));
    } else {
      _builder.setLoc(mlir::UnknownLoc::get(_builder.getContext()));
    }
  }

  void addGlobalState(std::string_view name, mlir::MemRefType argType);

  template <typename OpTy, typename... Args> OpTy createOp(Args &&...args) {
    return createOp<OpTy>(_main_graph, std::forward<Args>(args)...);
  }

  virtual TensorId addValue(const mlir::Value &value);

  mlir::Value findValue(TensorId tensor) const;

  // Update the MLIR value associated to a tensor id.
  // This is needed when an inplace view op changes the
  // underlying type (e.g. torch.select_) or when a tensor
  // was created by a graph and then used by another one:
  // the tensor would be an op output in the first graph
  // and a graph input in the second.
  void updateTensor(TensorId id, mlir::Value new_value);

  bool allOpsCanBeLoweredToPoplar() const;

protected:
  struct Graph {
    mlir::func::FuncOp graph;
    mlir::Block::iterator epilog_start;
    // When a new op is added to the main graph using createOp we check and
    // store whether or not there is an actual handler for this op. (Some ops
    // will have been added with only shape inference and no implementation, in
    // which case we won't be able to lower them later on).
    bool all_ops_can_be_lowered{true};

    Graph() = default;
    explicit Graph(mlir::func::FuncOp func) : graph(func) {
      // Add an entry block.
      graph.addEntryBlock();

      epilog_start = graph.front().end();
    }
  };
  // Remove all the ops in the main graph but do not reset
  // the tensor IDs. Any valid tensor ID passed to findValue()
  // will now return an empty Value.
  void resetMainGraph();

  Graph createSubGraph(const std::string &name);

  // Create a new op of class OpTy, possibly casting operands specified by args.
  template <typename OpTy, typename... Args>
  OpTy createOp(Graph &graph, mlir::Block::iterator insert_before,
                Args &&...args) {
    _builder.setInsertionPoint(&graph.graph.front(), insert_before);

    if constexpr (isImplicitCastingOp<OpTy>()) {
      mlir::Type promote_to = getCastToType<OpTy>(context, args...);
      OpTy op = implicitCastAndCreate<OpTy>(_builder, promote_to,
                                            std::forward<Args>(args)...);
      return op;
    }

    OpTy op = _builder.create<OpTy>(std::forward<Args>(args)...);

    graph.all_ops_can_be_lowered &=
        !OpTy::template hasTrait<mlir::OpTrait::NotImplementedOp>();
    return op;
  }

  template <typename OpTy, typename... Args>
  OpTy createOp(Graph &graph, Args &&...args) {
    return createOp<OpTy>(graph, graph.epilog_start,
                          std::forward<Args>(args)...);
  }

  template <typename OpTy, typename... Args>
  OpTy createOpInEpilogue(Graph &graph, Args &&...args) {
    auto new_op = createOp<OpTy>(graph, graph.graph.front().end(),
                                 std::forward<Args>(args)...);
    // NOTE: this doesn't work if the added operation does any casting
    if (graph.epilog_start == graph.graph.front().end()) {
      graph.epilog_start = mlir::Block::iterator(new_op);
    }
    return new_op;
  }

  llvm::DenseMap<mlir::Value, TensorId> getValueMappings() const;

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

  // Options to use in the compiler
  const poptorch::CompilerOptions *_compiler_options;
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_IMLIR_COMPILER_HPP_
