// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_HELPER_HPP_
#define POPTORCH_COMPILER_HELPER_HPP_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Value.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poplar/DataStream.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include "dialect/PoptorchDialect.hpp"

namespace model_runtime {
class Device;
}

namespace poptorch_ir {

enum Programs { MainGraph = 0, WeightsToDevice = 1, WeightsToHost = 2 };

struct CompilerContext {
  explicit CompilerContext(poplar::Graph &g, poplar::program::Sequence &s)
      : graph(g), seq(s) {
    poplin::addCodelets(g);
    popnn::addCodelets(g);
    popops::addCodelets(g);
    poprand::addCodelets(g);
  }

  poplar::Graph &graph;

  // The active sequence.
  poplar::program::Sequence &seq;

  // All poplar programs.
  std::array<poplar::program::Program, 3> programs;

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

  static poplar::Type poplarTypeOf(mlir::Type elementType);

  // Map the SSA to the corresponding Poplar tensor.
  // update_if_present: if the SSA is already present in the map then update
  // it to point at this new tensor.
  void addTensor(const mlir::Value &value, const poplar::Tensor &tensor,
                 bool update_if_present = false);

private:
  // Map of SSA->Poplar Tensors.
  llvm::DenseMap<mlir::Value, poplar::Tensor> _tensors;

  // Persistent seed to use for RNG functions.
  //
  // NOTE: This is a temporary workaround while TODO(T51096) remains unresolved,
  //       to handle loading, saving & restoring of the seed.
  std::optional<poplar::Tensor> _randomSeed;
};

poplar::Tensor reshapeToMlirShape(const poplar::Tensor &src,
                                  mlir::Type mlirType);

poplar::Type elementTypeFromMLIR(mlir::Type elementType);

std::shared_ptr<model_runtime::Device> getDevice();

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

  auto tile_num =
      (context.graph_const_count++) % context.graph.getTarget().getNumTiles();
  context.graph.setTileMapping(constant, tile_num);
  return constant;
}

// TODO(T60724) Not currently used, but will be useful when we add support for
// constant tensors.
template <typename T>
poplar::Tensor createConstantTensor(CompilerContext &context, poplar::Type type,
                                    const std::vector<uint64_t> &shape,
                                    const std::vector<T> &values) {
  poplar::Tensor constant =
      context.graph.addConstant<T>(type, shape, poplar::ArrayRef<T>(values));
  auto tile_num =
      (context.graph_const_count++) % context.graph.getTarget().getNumTiles();
  context.graph.setTileMapping(constant, tile_num);
  return constant;
}

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_HELPER_HPP_
