// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef SOURCE_POPART_CANONICALIZATION_UTILS_H
#define SOURCE_POPART_CANONICALIZATION_UTILS_H
#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <vector>

namespace poptorch {

using SymbolHandler =
    std::function<torch::jit::Node *(torch::jit::Graph *, torch::jit::Node *)>;

bool registerHandler(c10::Symbol symbol, const SymbolHandler &handler);

// Needed to end the recursion of registerHandlers
static bool __attribute__((unused)) registerHandlers() { return true; }

template <typename... OtherHandlers>
bool registerHandlers(c10::Symbol symbol, const SymbolHandler &handler,
                      OtherHandlers... handlers) {
  return registerHandler(symbol, handler) && registerHandlers(handlers...);
}

// Return a pointer to a handler if one is registered for this kind of node or
// an empty std::function otherwise.
SymbolHandler getHandler(torch::jit::Node *node);

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

// This handles the case of both `prim::ListConstruct`
// and 'prim::Constant[value=[x, y, z]]'.
template <typename T> std::vector<T> handleList(torch::jit::Node *node);

template <typename T>
std::vector<T> handleListConstruct(torch::jit::Node *node);

std::vector<torch::jit::Value *> handleTensorList(torch::jit::Node *node);

template <typename T> std::optional<T> handleConstant(torch::jit::Node *node);

// Some operations take in an optional tensor. A "none" constant is passed in to
// mark a tensor which is not there.
bool isNone(torch::jit::Node *node);

std::int64_t handleDimensionParam(torch::jit::Node *node, int index);

// Turn a prim::Constant scalar input into a popart graph level scalar constant.
torch::jit::Node *createIRConstant(torch::jit::Graph *graph,
                                   torch::jit::Value *value);

// Do not cast the operand.
torch::jit::Value *handleParamOrConstantNoCast(torch::jit::Graph *graph,
                                               torch::jit::Value *operand);

// Both pytorch and popart represent reduce as an enum but with different
// values.
std::int32_t convertReduceToPopart(std::int32_t pytorchReduce);

void markNodeForDeletion(torch::jit::Node *node);
bool isMarkedForDeletion(torch::jit::Node *node);

void replaceOutputUse(torch::jit::Value *old_val, torch::jit::Value *new_val);
void replaceOutputUse(torch::jit::Node *oldNode, torch::jit::Node *new_node,
                      std::uint64_t outputIdx);
} // namespace poptorch

#endif // SOURCE_POPART_CANONICALIZATION_UTILS_H
