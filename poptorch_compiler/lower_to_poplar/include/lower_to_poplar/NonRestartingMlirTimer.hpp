// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef LOWER_TO_POPLAR_NON_RESTARTING_MLIR_TIMER_HPP_
#define LOWER_TO_POPLAR_NON_RESTARTING_MLIR_TIMER_HPP_

#include <memory>

namespace mlir {
class Timer;
class TimingScope;
} // namespace mlir

namespace poptorch_ir {

// A wrapper for MlirTimer which stops it restarting when start is called.
class NonRestartingMlirTimer {
public:
  explicit NonRestartingMlirTimer(mlir::Timer &&timer);

  // Start the timer unless it is running already.
  void start();

  // Stop the timer (safe to call if it is not running).
  void stop();

  // Returns a timing scope on a nested timer: this allows for the usual
  // RAII timing of the nest.
  mlir::TimingScope nestAndScope(const char *name);

private:
  std::shared_ptr<bool> _running;
  std::shared_ptr<mlir::Timer> _timer;
};

} // namespace poptorch_ir

#endif // LOWER_TO_POPLAR_NON_RESTARTING_MLIR_TIMER_HPP_
