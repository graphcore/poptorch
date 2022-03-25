// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_HELPER_HPP_
#define POPTORCH_COMPILER_HELPER_HPP_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Value.h>

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poplar/DataStream.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

enum Programs { MainGraph = 0, WeightsToDevice = 1, WeightsToHost = 2 };

struct CompilerContext {
  explicit CompilerContext(poplar::Graph &g) : graph(g) {}

  poplar::Graph &graph;

  // The active sequence.
  poplar::program::Sequence seq;

  // All poplar programs.
  std::array<poplar::program::Program, 3> programs;

  // Map of SSA->Poplar Tensors.
  llvm::DenseMap<mlir::Value, poplar::Tensor> tensors;

  // Handles to their datastreams.
  std::unordered_map<std::string, poplar::DataStream> streams;

  poplar::Tensor fromSsa(mlir::Value value);

  std::vector<poplar::Tensor> fromSsa(mlir::ValueRange value_range);

  std::int64_t graph_const_count = 0;

  // Get a seed to use for RNG functions.
  //
  // If no seed is used (ie. if left `nullptr` in functions that use it), can
  // get strange results from `poprand` functions.
  //
  // NOTE: This is a temporary workaround while TODO(T51096) remains unresolved,
  //       to handle loading, saving & restoring of the seed.
  poplar::Tensor &getRandomSeed();

private:
  // Persistent seed to use for RNG functions.
  //
  // NOTE: This is a temporary workaround while TODO(T51096) remains unresolved,
  //       to handle loading, saving & restoring of the seed.
  std::optional<poplar::Tensor> _randomSeed;
};

template <typename T>
std::vector<T> convertFloatArray(const mlir::ArrayAttr &array) {
  std::vector<T> output;

  for (mlir::Attribute elem : array) {
    output.push_back(
        static_cast<T>(elem.cast<mlir::FloatAttr>().getValueAsDouble()));
  }

  return output;
}

template <typename T>
std::vector<T> convertIntArray(const mlir::ArrayAttr &array) {
  std::vector<T> output;

  for (mlir::Attribute elem : array) {
    output.push_back(static_cast<T>(elem.cast<mlir::IntegerAttr>().getUInt()));
  }

  return output;
}

template <typename T>
poplar::Tensor createConstant(CompilerContext &context, poplar::Type type,
                              const std::vector<uint64_t> &shape,
                              const T &value) {
  poplar::Tensor constant = context.graph.addConstant<T>(type, shape, value);
  context.graph.setTileMapping(constant, context.graph_const_count++);
  return constant;
}

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_HELPER_HPP_
