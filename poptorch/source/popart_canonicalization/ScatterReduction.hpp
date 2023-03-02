// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef SCATTER_REDUCTION_H
#define SCATTER_REDUCTION_H

#include <cstdint>

namespace torch {
namespace jit {
class Node;
} // namespace jit
} // namespace torch

namespace poptorch {

enum class ScatterReduction { Sum = 0, Max, Min, Mul, None, Mean };

std::int32_t getReductionMethod(torch::jit::Node *node);
float getReductionInitValue(std::int32_t reduce);

} // namespace poptorch

#endif
