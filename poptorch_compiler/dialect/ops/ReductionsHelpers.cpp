// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "dialect/ops/ReductionsHelpers.hpp"
#include "dialect/Helpers.hpp"

#include "mlir/IR/Dialect.h"
#include <numeric>

namespace poptorch_ir {
llvm::SmallVector<std::int64_t, 4>
inferShapeOfReductionOutput(std::vector<std::int64_t> dims,
                            llvm::ArrayRef<std::int64_t> in_shape,
                            bool keepdim) {
  llvm::SmallVector<std::int64_t, 4> shape{in_shape.begin(), in_shape.end()};

  // Optimization for when we are reducing over all the dimensions or the input
  // is a scalar
  if (in_shape.size() == dims.size() || in_shape.empty()) {
    if (!keepdim) {
      return {};
    }
    std::fill(shape.begin(), shape.end(), 1);
    return shape;
  }

  // Sort in descending order so erasing elements doesn't invalidate future
  // erases
  std::sort(dims.begin(), dims.end(), std::greater<>{});

  // Flatten those dims.
  for (std::int64_t const dim : dims) {
    // Dim reduced to 1 or zero depending on keep dim.
    if (keepdim) {
      shape[dim] = 1;
    } else {
      shape.erase(shape.begin() + dim);
    }
  }

  return shape;
}

std::vector<std::int64_t>
parseReductionDimArgument(const std::vector<std::int64_t> &input_dim,
                          std::size_t input_shape_size) {
  auto dims = convertToPositiveDim(input_dim, input_shape_size);

  // NOTE: an empty list of dimensions means we're reducing over all the
  // dimensions
  if (dims.empty()) {
    dims.resize(input_shape_size);
    std::iota(dims.begin(), dims.end(), 0);
  }

  // If we are dealing with a scalar value don't reduce at all
  if (input_shape_size == 0) {
    dims.clear();
  }

  return dims;
}
} // namespace poptorch_ir
