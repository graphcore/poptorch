// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <model_runtime/DeviceManager.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

#include "CompilerHelpers.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

CompilerContext::CompilerContext(poplar::Graph &g, poplar::program::Sequence &s)
    : graph(g), seq(s) {
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  poprand::addCodelets(graph);
}

poplar::Type elementTypeFromMLIR(mlir::Type elementType) {
  if (elementType.isF16()) {
    return poplar::HALF;
  }
  if (elementType.isF32()) {
    return poplar::FLOAT;
  }
  if (elementType.isUnsignedInteger(8)) {
    return poplar::UNSIGNED_CHAR;
  }
  if (elementType.isUnsignedInteger(16)) {
    return poplar::UNSIGNED_SHORT;
  }
  if (elementType.isUnsignedInteger(32) || elementType.isUnsignedInteger(64)) {
    return poplar::UNSIGNED_INT;
  }
  // We use isInteger from here onwards to capture both
  // isSignedInteger and isSignlessInteger
  if (elementType.isInteger(1)) {
    return poplar::BOOL;
  }
  if (elementType.isInteger(8)) {
    return poplar::SIGNED_CHAR;
  }
  if (elementType.isInteger(16)) {
    return poplar::SHORT;
  }
  if (elementType.isInteger(32) || elementType.isInteger(64)) {
    return poplar::INT;
  }
  ERROR("Unsupported MLIR type");

  return poplar::FLOAT;
}
} // namespace poptorch_ir
