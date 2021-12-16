// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_TRANSFORMS_PASS_UTILS_HPP_
#define POPTORCH_TRANSFORMS_PASS_UTILS_HPP_

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/OpDefinition.h>
#include <string>

namespace poptorch_ir {

inline std::string mlirOpToStr(mlir::OpState &op) {
  std::string str;
  llvm::raw_string_ostream ostream(str);
  ostream << op;
  return str;
}

} // namespace poptorch_ir

#endif // POPTORCH_TRANSFORMS_PASS_UTILS_HPP_
