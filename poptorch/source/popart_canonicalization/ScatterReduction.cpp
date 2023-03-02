// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include "ScatterReduction.hpp"
#include "PopartCanonicalizationUtils.hpp"

namespace poptorch {

std::int32_t getReductionMethod(torch::jit::Node *node) {
  const auto reduce = constantToString(node);
  if (reduce == "sum" || reduce == "add") {
    return static_cast<std::int32_t>(ScatterReduction::Sum);
  }
  if (reduce == "amax") {
    return static_cast<std::int32_t>(ScatterReduction::Max);
  }
  if (reduce == "amin") {
    return static_cast<std::int32_t>(ScatterReduction::Min);
  }
  if (reduce == "mean") {
    return static_cast<std::int32_t>(ScatterReduction::Mean);
  }
  if (reduce == "prod" || reduce == "multiply") {
    return static_cast<std::int32_t>(ScatterReduction::Mul);
  }

  ERROR("Unsupported reduction type for scatter_reduce: " << reduce);
}

float getReductionInitValue(std::int32_t reduce) {
  float init_val;
  switch (reduce) {
  case static_cast<std::int32_t>(ScatterReduction::Sum):
  case static_cast<std::int32_t>(ScatterReduction::Mean):
    init_val = 0.0;
    break;
  case static_cast<std::int32_t>(ScatterReduction::Mul):
    init_val = 1.0;
    break;
  case static_cast<std::int32_t>(ScatterReduction::Max):
    init_val = -std::numeric_limits<float>::infinity();
    break;
  case static_cast<std::int32_t>(ScatterReduction::Min):
    init_val = std::numeric_limits<float>::infinity();
    break;
  default:
    ERROR("Unsupported reduction type for scatter_reduce: " << reduce);
    break;
  }
  return init_val;
}

} // namespace poptorch
