// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_POPIT_CONTEXT_HPP_
#define POPTORCH_POPIT_CONTEXT_HPP_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinOps.h>

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include <popit/Device.hpp>
#include <popit/functions.hpp>
#include <popit/popit.hpp>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace mlir {
class Value;
class FuncOp;
} // namespace mlir

namespace poptorch_ir {

class PopitMemPtr : public std::shared_ptr<popit::Mem_t> {
public:
  explicit PopitMemPtr(popit::Mem_t *ptr)
      : std::shared_ptr<popit::Mem_t>(ptr, popit::free) {}
};

class PopitContext {
public:
  PopitContext();
  ~PopitContext();

  // Map of all the PopIT allocated tensors.
  std::unordered_map<TensorId, PopitMemPtr> tensors;

  std::unique_ptr<popit::Session_t, void (*)(popit::Session *)> session;
  // We need to keep around the device used by the session or it will segfault.
  popit::Device device;

  // These attributes get populated by LowerToPopit
  popit::FunctionId_t popit_fn;
  std::vector<mlir::Value> inputs;
  std::vector<TensorId> output_ids;
};

} // namespace poptorch_ir

#endif // POPTORCH_POPIT_CONTEXT_HPP_
