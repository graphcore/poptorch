// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <mlir/IR/BuiltinTypes.h>
#include <model_runtime/DeviceManager.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/Fill.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include "CompilerHelpers.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

CompilerContext::CompilerContext(poplar::Graph &g, poplar::program::Sequence &s)
    : graph(g), seq(s) {
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  poprand::addCodelets(graph);
}

poplar::Type elementTypeFromMLIR(mlir::Type elementType) {
  if (elementType.isF16()) {
    return poplar::HALF;
  }
  if (elementType.isF32()) {
    return poplar::FLOAT;
  }
  if (elementType.isUnsignedInteger(8)) {
    return poplar::UNSIGNED_CHAR;
  }
  if (elementType.isUnsignedInteger(16)) {
    return poplar::UNSIGNED_SHORT;
  }
  if (elementType.isUnsignedInteger(32) || elementType.isUnsignedInteger(64)) {
    return poplar::UNSIGNED_INT;
  }
  // We use isInteger from here onwards to capture both
  // isSignedInteger and isSignlessInteger
  if (elementType.isInteger(1)) {
    return poplar::BOOL;
  }
  if (elementType.isInteger(8)) {
    return poplar::SIGNED_CHAR;
  }
  if (elementType.isInteger(16)) {
    return poplar::SHORT;
  }
  if (elementType.isInteger(32) || elementType.isInteger(64)) {
    return poplar::INT;
  }
  ERROR("Unsupported MLIR type");

  return poplar::FLOAT;
}

namespace {
bool waitIfIpuIsUnavailable() {
  bool wait = false;
  if (const char *env_wait_for_ipu = std::getenv("POPTORCH_WAIT_FOR_IPU")) {
    wait = std::stoi(env_wait_for_ipu) != 0;
    poptorch::logging::info(
        "From POPTORCH_WAIT_FOR_IPU environment variable: If no IPU "
        "is available: {}",
        wait ? "Wait" : "Fail & exit");
  }
  return wait;
}
} // namespace

std::shared_ptr<model_runtime::Device> getDevice() {
  std::shared_ptr<model_runtime::Device> device;
  model_runtime::DeviceManager manager;

  bool model_enabled = false;

  // Run on model if the env var is set.
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (model_enabled) {
      device = manager.createIpuModelDevice(1);
    }
  }
  if (const char *env_use_model = std::getenv("POPTORCH_SMALL_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (!device && model_enabled) {
      device = manager.createSmallIpuModelDevice(1);
    }
  }
  // Run on an actual device.
  if (!device) {
    using WaitStrategy = model_runtime::DeviceWaitStrategy;
    model_runtime::DeviceWaitConfig wait_config;
    wait_config.strategy = waitIfIpuIsUnavailable() ? WaitStrategy::WAIT_FOREVER
                                                    : WaitStrategy::NO_WAIT;
    device = manager.getDevice(1, {}, wait_config);
  }
  ERROR_ON_MSG(!device, "Failed to acquire a device");
  return device;
}

poplar::Tensor reshapeToMLIRShape(const poplar::Tensor &src,
                                  mlir::Type mlirType) {
  return src.reshape(processType(mlirType).shape);
}

poplar::Type CompilerContext::poplarTypeOf(mlir::Type elementType) {
  return elementTypeFromMLIR(elementType);
}

PoplarTypePair processType(mlir::Type mlirType) {
  // Turn it into a ranked tensor.
  mlir::RankedTensorType tensor_type = mlirType.cast<mlir::RankedTensorType>();
  mlir::Type element_type = tensor_type.getElementType();

  // Extract the element type of the tensor.
  poplar::Type type = elementTypeFromMLIR(element_type);

  // Convert the dimensions into something poplar understands.
  std::vector<std::size_t> dims;
  for (int64_t dim : tensor_type.getShape()) {
    dims.push_back(dim);
  }

  return {type, dims};
}

std::string toString(const std::vector<std::size_t> &shape,
                     const poplar::Type &type) {
  std::stringstream ss;
  ss << type << "[";
  std::string sep{};
  for (const auto &s : shape) {
    ss << sep << s;
    sep = ", ";
  }
  ss << "]";
  return ss.str();
}

void CompilerContext::addTensor(const mlir::Value &value,
                                const poplar::Tensor &tensor,
                                bool update_if_present) {
  auto mlir_type = processType(value.getType());
  std::string mlir_shape = toString(mlir_type.shape, mlir_type.element_type);
  std::string poplar_shape = toString(tensor.shape(), tensor.elementType());
  ERROR_ON_MSG(mlir_shape != poplar_shape,
               "The shape of the Poplar tensor "
                   << poplar_shape
                   << " doesn't match the shape of the MLIR tensor it's "
                      "associated with: "
                   << mlir_shape << " for " << mlirToStr(value));

  if (update_if_present) {
    _tensors[value] = tensor;
  } else {
    auto res = _tensors.insert({value, tensor});
    ERROR_ON_MSG(!res.second,
                 "[Internal] Tensor already present for " << mlirToStr(value));
  }
}

void CompilerContext::addTensor(std::string_view symbol_name,
                                const poplar::Tensor &tensor,
                                bool update_if_present) {
  if (update_if_present) {
    _global_tensors[std::string(symbol_name)] = tensor;
  } else {
    auto res = _global_tensors.insert({std::string(symbol_name), tensor});
    ERROR_ON_MSG(!res.second,
                 "[Internal] Tensor already present for " << symbol_name);
  }
}

// Get the poplar tensor which corresponds to a specific value of MLIR.
poplar::Tensor CompilerContext::fromSymbol(std::string_view symbol_name,
                                           mlir::Type type) {
  auto itr = _global_tensors.find(std::string(symbol_name));
  if (itr != _global_tensors.end()) {
    return itr->second;
  }

  const PoplarTypePair tensor_type = processType(type);

  // Actually add the tensor to the graph.
  poplar::Tensor tensor =
      this->graph.addVariable(tensor_type.element_type, tensor_type.shape,
                              poplar::VariableMappingMethod::LINEAR);

  addTensor(symbol_name, tensor);
  return tensor;
}

// Get the poplar tensor which corresponds to a specific value of MLIR.
poplar::Tensor CompilerContext::fromSsa(mlir::Value value) {
  auto itr = _tensors.find(value);
  if (itr != _tensors.end()) {
    return itr->second;
  }

  const PoplarTypePair tensor_type = processType(value.getType());

  // Actually add the tensor to the graph.
  poplar::Tensor tensor =
      this->graph.addVariable(tensor_type.element_type, tensor_type.shape,
                              poplar::VariableMappingMethod::LINEAR);

  addTensor(value, tensor);
  return tensor;
}

std::vector<poplar::Tensor>
CompilerContext::fromSsa(mlir::ValueRange value_range) {
  std::vector<poplar::Tensor> poplar_tensors;

  for (mlir::Value value : value_range) {
    poplar_tensors.push_back(fromSsa(value));
  }

  return poplar_tensors;
}

poplar::Tensor &CompilerContext::getRandomSeed() {
  // NOTE: This mechanism is a temporary workaround while TODO(T51096) remains
  //       unresolved, to handle loading, saving & restoring of the seed.
  if (!_randomSeed) {
    _randomSeed = graph.addVariable(poplar::UNSIGNED_INT, {2},
                                    poplar::VariableMappingMethod::LINEAR);
    popops::fill(graph, *_randomSeed, seq, 42);
  }

  return *_randomSeed;
}

void CompilerContext::clearLocalData() { _tensors.clear(); }
} // namespace poptorch_ir
