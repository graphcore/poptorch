// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <numeric>
#include <utility>

#include "dialect/Helpers.hpp"
#include "dialect/ops/NormOpsHelpers.hpp"

#include "mlir/IR/Dialect.h"

#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {
std::tuple<size_t, size_t, std::vector<int64_t>>
checkAndGetLayerNormDimsOperandsAndSegs(
    std::vector<mlir::Value> &operands, std::vector<std::int32_t> &segments,
    const mlir::Value &input, const std::vector<std::int64_t> &normalized_shape,
    const mlir::Value &weight, const mlir::Value &bias) {

  ERROR_ON_MSG(normalized_shape.empty(),
               "normlized_shape must be at least 1-dimensional.");
  if (weight) {
    ERROR_ON(!bias);

    ERROR_ON_MSG(getShape(weight).size() != normalized_shape.size(),
                 "weight must be the same size as normalized_shape.");
    ERROR_ON_MSG(getShape(bias).size() != normalized_shape.size(),
                 "bias must be the same size as normalized_shape.");
    ERROR_ON_MSG(getElementType(weight) != getElementType(input),
                 "Type mismatch: input ("
                     << elementTypeToString(getElementType(input))
                     << ") != weight ("
                     << elementTypeToString(getElementType(weight)) << ")");

    ERROR_ON_MSG(getElementType(bias) != getElementType(input),
                 "Type mismatch: input ("
                     << elementTypeToString(getElementType(input))
                     << ") != bias ("
                     << elementTypeToString(getElementType(bias)) << ")");

    segments[operands.size()] = 1;
    operands.push_back(weight);

    segments[operands.size()] = 1;
    operands.push_back(bias);
  }

  auto input_shape = getShape(input);

  size_t skip_first = input_shape.size() - normalized_shape.size();
  if (input_shape.size() < normalized_shape.size()) {
    auto sliced_size = std::vector<int64_t>(input_shape.begin() + skip_first,
                                            input_shape.end());
    ERROR_ON_MSG(normalized_shape != sliced_size,
                 "normalized_shape is an invalid size");
  }

  auto m_dim =
      std::accumulate(input_shape.begin(), input_shape.begin() + skip_first,
                      static_cast<int64_t>(1), std::multiplies<int64_t>());

  auto n_dim =
      std::accumulate(input_shape.begin() + skip_first, input_shape.end(),
                      static_cast<int64_t>(1), std::multiplies<int64_t>());

  std::vector<int64_t> stat_shape(input_shape.begin(),
                                  input_shape.begin() + skip_first);

  for (unsigned int i = 0; i < normalized_shape.size(); i++) {
    stat_shape.emplace_back(1);
  }

  return {m_dim, n_dim, stat_shape};
}

} // namespace poptorch_ir
