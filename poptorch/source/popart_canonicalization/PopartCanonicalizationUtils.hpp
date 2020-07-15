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

std::vector<std::int64_t> shapeFromTensor(torch::jit::Value *value);

// This handles the case of both `prim::ListConstruct`
// and 'prim::Constant[value=[x, y, z]]'.
template <typename T> std::vector<T> handleList(torch::jit::Node *node);

template <typename T>
std::vector<T> handleListConstruct(torch::jit::Node *node);

template <typename T> std::optional<T> handleConstant(torch::jit::Node *node);
} // namespace poptorch

#endif // SOURCE_POPART_CANONICALIZATION_UTILS_H
