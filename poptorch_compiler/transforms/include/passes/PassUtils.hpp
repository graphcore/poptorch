// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_TRANSFORMS_PASS_UTILS_HPP_
#define POPTORCH_TRANSFORMS_PASS_UTILS_HPP_

#include <llvm/Support/raw_ostream.h>
#include <string>

namespace poptorch_ir {

// Convert an MLIR OpState or Operation to a string
template <class T> std::string mlirOpToStr(T &op) {
  std::string str;
  llvm::raw_string_ostream ostream(str);
  ostream << op;
  return str;
}

} // namespace poptorch_ir

#endif // POPTORCH_TRANSFORMS_PASS_UTILS_HPP_
