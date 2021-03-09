// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef SOURCE_POPART_CANONICALIZATION_UTILS_H
#define SOURCE_POPART_CANONICALIZATION_UTILS_H
#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <string>
#include <vector>

#include "poptorch_logging/Error.hpp"

namespace poptorch {

using SymbolHandler =
    std::function<torch::jit::Node *(torch::jit::Graph *, torch::jit::Node *)>;

bool registerHandler(c10::Symbol symbol, const SymbolHandler &handler);

std::vector<std::int64_t>
reduceHelperDimensionCreator(torch::jit::Value *value);

static inline std::vector<std::int64_t>
reduceHelperDimensionCreator(torch::jit::Value *value,
                             const std::vector<std::int64_t> &axes) {
  if (!axes.empty()) {
    return axes;
  }
  return reduceHelperDimensionCreator(value);
}

// Return a pointer to a handler if one is registered for this kind of node or
// an empty std::function otherwise.
SymbolHandler getHandler(torch::jit::NodeKind kind);

// Returns true if all inputs are Bools
bool allInputsBool(torch::jit::Node *node, int ignore_input = -1);

// Returns true if all inputs are Bools
bool allInputsInteger(torch::jit::Node *node, int ignore_input = -1);

// Get the tensor shape as a vector of ints.
std::vector<std::int64_t> shapeFromTensor(torch::jit::Value *value);

// Get the tensor shape and add it to the IR as a constant primative.
torch::jit::Value *shapeFromTensorAsIR(torch::jit::Graph *graph,
                                       torch::jit::Value *value);

// Get the scalar type of this tensor.
at::ScalarType getNodeScalarType(torch::jit::Value *tensor);

torch::jit::Value *
intVectorToIrConstant(torch::jit::Graph *graph,
                      const std::vector<std::int64_t> &shape);

std::vector<torch::jit::Value *> handleTensorList(torch::jit::Node *node);

// Returns true if the value is a constant of exactly unity (1)
bool hasUnityValue(torch::jit::Value *value);

// Some operations take in an optional tensor. A "none" constant is passed in to
// mark a tensor which is not there.
bool isNone(torch::jit::Node *node);
bool isNone(const torch::jit::Value *value);

std::int64_t handleDimensionParam(torch::jit::Value *value,
                                  const c10::TensorTypePtr &as_tensor);

bool isAnyConstant(torch::jit::Node *node);

bool isFloatingPointConstant(torch::jit::Node *node);

bool isTensorConstant(torch::jit::Node *node);

// Force a constant to be a float: this is appropriate if required for popart
// (onnx); e.g. Gemm alpha and beta are always floats
float constantToFloat(torch::jit::Node *node);

// Force a constant to be a long constant by casting.
// This is appropriate if required for popart (onnx)
// e.g. TopK takes int64 indices as a tensor.
torch::jit::Node *constantToLongConstant(torch::jit::Node *node);

// Force a constant to be an int: this is appropriate if required for popart
// (onnx)
std::int32_t constantToInt(torch::jit::Node *node);

// Force a constant to be a long: this is appropriate if required for popart
// (onnx) e.g. Slice takes int64 indices
std::int64_t constantToLong(torch::jit::Node *node);

// Forces a ListConstruct to be a vector of int64_ts
std::vector<std::int64_t> constantToLongVec(torch::jit::Node *node);

// Extract a boolean from a constant containing one (encoded as an int32_t)
bool constantToBool(torch::jit::Node *node);

// Extracts a string from a constant containing a string
std::string constantToString(torch::jit::Node *node);

// Forces a ListConstuct into a vector of the given type
template <typename T, typename ExtractFunc>
std::vector<T> constantListToVec(torch::jit::Node *node,
                                 ExtractFunc &&constantExtractFunc) {
  ERROR_ON(node->kind() != c10::prim::ListConstruct);

  auto node_inputs = node->inputs();
  std::vector<T> result;
  result.reserve(node_inputs.size());
  for (torch::jit::Value *value : node_inputs) {
    result.push_back(constantExtractFunc(value->node()));
  }

  return result;
}

// Both pytorch and popart represent reduce as an enum but with different
// values.
std::int32_t convertReduceToPopart(std::int32_t pytorchReduce);

void markNodeForDeletion(torch::jit::Node *node);
bool isMarkedForDeletion(torch::jit::Node *node);

void replaceOutputUse(torch::jit::Value *old_val, torch::jit::Value *new_val);
void replaceOutputUse(torch::jit::Node *oldNode, torch::jit::Node *new_node,
                      std::uint64_t outputIdx);

bool attributeEqual(torch::jit::Node *a, torch::jit::Node *b,
                    c10::Symbol attrb);

} // namespace poptorch

#endif // SOURCE_POPART_CANONICALIZATION_UTILS_H
