// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_POPIT_CONTEXT_HPP_
#define POPTORCH_POPIT_CONTEXT_HPP_

#include <llvm/ADT/DenseMap.h>

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include <popit/functions.hpp>
#include <popit/popit.hpp>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace mlir {
class Value;
} // namespace mlir

namespace poptorch_ir {

class PopitMemPtr : public std::shared_ptr<popitMem_t> {
public:
  explicit PopitMemPtr(popitMem_t *ptr)
      : std::shared_ptr<popitMem_t>(ptr, popitFree) {}
};

class PopitContext {
public:
  PopitContext();
  ~PopitContext();

  // Map of all the PopIT allocated tensors.
  std::unordered_map<TensorId, PopitMemPtr> tensors;

  std::unique_ptr<popitSession_t, void (*)(popitSession *)> session;

  // These attributes get populated by LowerToPopit
  popitFunctionId_t popit_fn;
  std::vector<mlir::Value> inputs;
  std::deque<mlir::Value> outputs;
  std::vector<TensorId> output_ids;
};

} // namespace poptorch_ir

#endif // POPTORCH_POPIT_CONTEXT_HPP_
