// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <poptorch_logging/Error.hpp>

#include "dialect/Poptorch.hpp"

namespace poptorch_ir {

::mlir::LogicalResult copy_::verify() {
  if (self().getType() != src().getType()) {
    ERROR("The source and destination of an inplace copy must have the same "
          "shape and dtype.");
  }
  return mlir::success();
}

} // namespace poptorch_ir
