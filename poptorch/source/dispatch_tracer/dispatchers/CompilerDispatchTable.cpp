// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <vector>

#include "poptorch_logging/Error.hpp"

#include "../../PoptorchSymbols.hpp"
#include "MlirDispatch.hpp"

/*
 * This file implements the hooks the compiler uses to interface with the
 * PopTorch compiler.
 *
 * The indended operational flow is as follows: Aten operation is encountered.
 *  If it is a directly support MLIR op:
 *    We unpack all the PyTorch arguments into MLIR arguments and call the
 *    autogenerated builder.
 *
 *    These are defined in OpSupport.yml
 *
 *  If it is not directly supported in MLIR, then it will go through our normal
 *  canonicalisation process. In this case the dispatch will call another set of
 *  autogenerated functions which unpack the torch::jit::Node in the same way
 *  LowerToPopart does. They use a reduced OP_DEF format which is:
 *    OP_DECL(dynamicadd, dynamicadd,  ARG(INT_VEC,axes) ARG(INT_VEC,sizes))
 *  Unlike MLIR these must specify the arguments (as MLIR doesn't know the JIT
 *  attribute names). However it is still far less than the original lower to
 *  PopART set. Once we have full 1:1 coverage we might want to consider sharing
 *  the same list.
 *
 * JIT fallback ops are defined in: PopartAPISupportedOps.h.inc
 */

namespace poptorch {

/*
 * The first dispatch table. This is the one which maps from an aten operation
 * onto the above table.
 */
void MLIRDispatch::generateDispatchTable() { // NOLINT
  // The generated mapping of PyTorch/Aten -> MLIR functions.
  // Each of functions when passed the PyTorch function arguments will extract
  // and translate all tensors/scalars into an MLIR representation from the
  // given PyTorch and then call the correct MLIR builder function.

  // It is generated by the generate scripts in
  // `poptorch/source/dispatch_tracer/scripts` from OpSupport.yml.
  _direct_dispatch_lookup = {
#include "AtenToMlirDispatch.inc"
  };

  /*
   * The second dispatch table is used to dispatch from the current TracingV1
   * like PopART/JIT IR nodes we create. This is so we can still support the
   * normal handler path.
   */
}

} // namespace poptorch
