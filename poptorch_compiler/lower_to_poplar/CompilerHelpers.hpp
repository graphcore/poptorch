// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_HELPER_HPP_
#define POPTORCH_COMPILER_HELPER_HPP_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

#include "dialect/Helpers.hpp"
#include "dialect/PoptorchDialect.hpp"

namespace poplar {
class DataStream;
} // namespace poplar

namespace poptorch_ir {

enum Programs { MainGraph = 0, WeightsToDevice = 1, WeightsToHost = 2 };

struct CompilerContext {
  explicit CompilerContext(poplar::Graph &g, poplar::program::Sequence &s);

  poplar::Graph &graph;

  // The active sequence.
  poplar::program::Sequence &seq;

  // All poplar programs.
  std::array<poplar::program::Program, 3> programs;

  // Handles to their datastreams.
  std::unordered_map<std::string, poplar::DataStream> streams;

  poplar::Tensor fromSsa(mlir::Value value);

  std::vector<poplar::Tensor> fromSsa(mlir::ValueRange value_range);

  poplar::Tensor fromSymbol(std::string_view symbol_name, mlir::Type type);

  std::int64_t graph_const_count = 0;

  // Get a seed to use for RNG functions.
  //
  // If no seed is used (ie. if left `nullptr` in functions that use it), can
  // get strange results from `poprand` functions.
  //
  // NOTE: This is a temporary workaround while TODO(T51096) remains unresolved,
  //       to handle loading, saving & restoring of the seed.
  poplar::Tensor &getRandomSeed();

  static poplar::Type poplarTypeOf(mlir::Type elementType);

  // Map the SSA to the corresponding Poplar tensor.
  // update_if_present: if the SSA is already present in the map then update
  // it to point at this new tensor.
  void addTensor(const mlir::Value &value, const poplar::Tensor &tensor,
                 bool update_if_present = false);

  void addTensor(std::string_view symbol_name, const poplar::Tensor &tensor,
                 bool update_if_present = false);

  void clearLocalData();

private:
  // Local map of SSA->Poplar Tensors.
  llvm::DenseMap<mlir::Value, poplar::Tensor> _tensors;

  // Local map of symbol strings -> Poplar Tensors
  std::unordered_map<std::string, poplar::Tensor> _global_tensors;

  // Persistent seed to use for RNG functions.
  //
  // NOTE: This is a temporary workaround while TODO(T51096) remains unresolved,
  //       to handle loading, saving & restoring of the seed.
  std::optional<poplar::Tensor> _randomSeed;
};

poplar::Tensor reshapeToMLIRShape(const poplar::Tensor &src,
                                  mlir::Type mlirType);

poplar::Type elementTypeFromMLIR(mlir::Type elementType);

std::string toString(const std::vector<std::size_t> &shape,
                     const poplar::Type &type);

// Poplar keeps the size and shape as two distinct concepts.
struct PoplarTypePair {
  poplar::Type element_type;
  std::vector<std::size_t> shape;
};

PoplarTypePair processType(mlir::Type mlirType);

template <typename T>
std::vector<T> convertFloatArray(const mlir::ArrayAttr &array) {
  std::vector<T> output;

  for (mlir::Attribute const elem : array) {
    output.push_back(
        static_cast<T>(elem.cast<mlir::FloatAttr>().getValueAsDouble()));
  }

  return output;
}

template <typename T>
std::vector<T> convertIntArray(const mlir::ArrayAttr &array) {
  std::vector<T> output;

  for (mlir::Attribute const elem : array) {
    output.push_back(static_cast<T>(elem.cast<mlir::IntegerAttr>().getUInt()));
  }

  return output;
}

template <typename T>
std::optional<std::vector<T>>
convertOptionalIntArray(const mlir::Optional<mlir::ArrayAttr> &array) {
  if (!array.has_value()) {
    return std::nullopt;
  }

  std::vector<T> output;
  for (mlir::Attribute const elem : *array) {
    output.push_back(static_cast<T>(elem.cast<mlir::IntegerAttr>().getUInt()));
  }

  return output;
}

template <typename T>
poplar::Tensor createConstant(CompilerContext &context, poplar::Type type,
                              const std::vector<uint64_t> &shape,
                              const T &value) {
  poplar::Tensor constant = context.graph.addConstant<T>(type, shape, value);
  const auto tile_num =
      (context.graph_const_count++) % context.graph.getTarget().getNumTiles();
  context.graph.setTileMapping(constant, tile_num);
  return constant;
}

template <typename T>
poplar::Tensor createConstantTensor(CompilerContext &context, poplar::Type type,
                                    const std::vector<uint64_t> &shape,
                                    const std::vector<T> &values) {
  poplar::Tensor constant =
      context.graph.addConstant<T>(type, shape, poplar::ArrayRef<T>(values));
  const auto tile_num =
      (context.graph_const_count++) % context.graph.getTarget().getNumTiles();
  context.graph.setTileMapping(constant, tile_num);
  return constant;
}

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_HELPER_HPP_
