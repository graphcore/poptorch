// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

namespace poptorch_ir {
class CompilerContext;
}

// TODO(T49565): Needed in LLVM-13
//#include "dialect/PoptorchDialect.cpp.inc"

// Easier to handle the tablegen include.
using namespace mlir;

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

#include "mlir/IR/OpImplementation.h"

// Include the operation definitions.
#define GET_OP_CLASSES
#include "dialect/Poptorch.cpp.inc"
#undef GET_OP_CLASSES
