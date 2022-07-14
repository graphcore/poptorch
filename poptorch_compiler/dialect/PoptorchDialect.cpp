// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {
struct CompilerContext;
}

#include "dialect/PoptorchDialect.cpp.inc"

// Easier to handle the tablegen include.
// Lint exception: Do not use namespace using-directives.
// Use using-declarations instead.
using namespace mlir; // NOLINT

namespace poptorch_ir {

std::vector<llvm::StringRef> convert(const std::vector<const char *> &strs) {
  std::vector<llvm::StringRef> vec;

  vec.reserve(strs.size());
  for (const char *str : strs) {
    vec.push_back(str);
  }

  return vec;
}

std::vector<int64_t> broadcast(const std::vector<int64_t> &lhs,
                               const std::vector<int64_t> &rhs,
                               size_t end_skip) {
  // Ensure lhs dims >= rhs dims or swap and process,
  if (lhs.size() < rhs.size()) {
    return broadcast(rhs, lhs);
  }

  auto lhs_itr = lhs.begin();
  auto rhs_itr = rhs.begin();

  // The rhs may have fewer dims.
  size_t missing_dims = lhs.size() - rhs.size();

  // The resolution happens from the trailing dimensions but the indices
  // are from leading dimensions.

  // If it's missing in rhs, copy the dimension from lhs.
  std::vector<int64_t> output_shape(lhs.size());
  std::copy(lhs_itr, lhs_itr + missing_dims, output_shape.begin());

  // Otherwise line up the trailing dimension and process.
  lhs_itr += missing_dims;

  for (size_t dim = missing_dims; dim < lhs.size() - end_skip;
       dim++, lhs_itr++, rhs_itr++) {
    size_t lhs_dim = *lhs_itr;
    size_t rhs_dim = *rhs_itr;

    if (lhs_dim == rhs_dim) {
      output_shape.at(dim) = lhs_dim;
    } else {
      if (lhs_dim == 1) {
        output_shape.at(dim) = rhs_dim;
      } else if (rhs_dim == 1) {
        output_shape.at(dim) = lhs_dim;
      } else {
        ERROR("The tensors cannot be broadcasted. See "
              "https://pytorch.org/docs/stable/notes/broadcasting.html for "
              "guidance on broadcasting.");
      }
    }
  }

  return output_shape;
}

#include "dialect/PoptorchInterfaces.cpp.inc"

void PoptorchDialect::initialize() {
  // Add the operations to the dialect.
  addOperations<
#define GET_OP_LIST
#include "dialect/Poptorch.cpp.inc"
#undef GET_OP_LIST
      >();
}
} // namespace poptorch_ir

// Include the operation definitions.
#define GET_OP_CLASSES
// Lint exception: file already included above
#include "dialect/Poptorch.cpp.inc" // NOLINT
#undef GET_OP_CLASSES
