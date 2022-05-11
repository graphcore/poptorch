// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_TRANSFORMS_PASS_UTILS_HPP_
#define POPTORCH_TRANSFORMS_PASS_UTILS_HPP_

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/OperationSupport.h>
#include <string>

namespace poptorch_ir {

// Convert an MLIR OpState or Operation to a string
template <class T> std::string mlirOpToStr(T &op) {
  std::string str;
  llvm::raw_string_ostream ostream(str);
  mlir::OpPrintingFlags flags{};
  // enableDebugInfo = add location() at the end of each line.
  // pretty = true -> Print the actual filename:line:col rather than loc0, loc1,
  // etc which are IDs in the mlir::SourceManager.
  flags.enableDebugInfo(/* prettyForm=*/true);
  op.print(ostream, flags);
  return str;
}

// Convert any MLIR object to string.
template <typename T> std::string mlirToStr(const T &obj) {
  std::string str;
  {
    llvm::raw_string_ostream ostream(str);
    ostream << obj;
  }
  return str;
}
} // namespace poptorch_ir

#endif // POPTORCH_TRANSFORMS_PASS_UTILS_HPP_
