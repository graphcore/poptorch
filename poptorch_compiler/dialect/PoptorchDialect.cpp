// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {
struct CompilerContext;
}

// TODO(T49565): Needed in LLVM-13
//#include "dialect/PoptorchDialect.cpp.inc"

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
