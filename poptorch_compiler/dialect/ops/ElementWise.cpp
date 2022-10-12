// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poptorch_logging/Error.hpp>

#include "dialect/Poptorch.hpp"

namespace poptorch_ir {

mlir::LogicalResult floor_divide::verify() {
  // emitWarning by default will try to print the op emitting the warning
  // However, before the op is printed it will be verified, causing an
  // infinite recursion here, so temporarily disable op printing on diagnistic.
  getContext()->printOpOnDiagnostic(false);
  emitWarning("floor_divide is deprecated. Use div with rounding_mode='trunc' "
              "for equivalent behaviour");
  getContext()->printOpOnDiagnostic(true);
  return mlir::success();
}

mlir::LogicalResult clampTensor::verify() {
  if (!min() && !max()) {
    ERROR("torch.clamp: At least one of 'min' or 'max' must not be None");
  }
  return mlir::success();
}

} // namespace poptorch_ir
