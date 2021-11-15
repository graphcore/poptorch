// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_POPLAR_EXECUTOR_HPP_
#define POPTORCH_LOWER_TO_POPLAR_POPLAR_EXECUTOR_HPP_

#include <memory>
#include <string>

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

  void init(mlir::ModuleOp module);
  void compile(poprithms::logging::ManualTimePartitionLogger &timer);
  void execute();

  void weightsToDevice();
  void weightsToHost();

  void connectStream(const std::string &string, void *ptr);

private:
  // Impl
  std::unique_ptr<detail::PoplarExecutableImpl> _impl;
};

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_POPLAR_EXECUTOR_HPP_
