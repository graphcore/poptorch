// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "poptorch_logging/Error.hpp"

#include "../CompilerHelpers.hpp"

namespace poptorch_ir {

void lstm_input::lowerToPoplar(CompilerContext &context) {
  context.fromSsa(this->input());
  ERROR("Not yet implemented");
}

} // namespace poptorch_ir
