// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_POPLAR_EXECUTOR_HPP_
#define POPTORCH_LOWER_TO_POPLAR_POPLAR_EXECUTOR_HPP_

#include <memory>
#include <string>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poprithms {
namespace logging {
class ManualTimePartitionLogger;
} // namespace logging.
} // namespace poprithms

namespace mlir {
class ModuleOp;
class TimingScope;
} // namespace mlir

namespace poptorch_ir {

namespace detail {
class PoplarExecutableImpl;
}

class PoplarExecutable {
public:
  explicit PoplarExecutable(mlir::ModuleOp module);
  ~PoplarExecutable();

  PoplarExecutable(PoplarExecutable &&other);

  operator bool() const { return static_cast<bool>(_impl); }

  void init(mlir::ModuleOp module);

  // Compile graph by running both PopTorch compiler passes and poplar
  // compilation.
  void compile(poprithms::logging::ManualTimePartitionLogger &timer);

  // Run graph on device.
  void execute();

  // Transfer weights from host to device
  void weightsToDevice();

  // Transfer weights from device to host
  void weightsToHost();

  // Connect to a poplar stream with a fixed location in memory.
  // Each time Poplar copies data to/from the named stream, it will read/write
  // to/from this memory locaiton.
  void connectStream(const std::string &string, void *ptr);
  void connectStream(const std::string &string, Buffer ptr);

private:
  // Impl
  std::unique_ptr<detail::PoplarExecutableImpl> _impl;
};

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_POPLAR_EXECUTOR_HPP_
