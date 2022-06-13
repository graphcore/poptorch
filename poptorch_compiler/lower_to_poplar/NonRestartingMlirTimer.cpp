// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/NonRestartingMlirTimer.hpp"

#include <mlir/Support/Timing.h>

namespace poptorch_ir {

NonRestartingMlirTimer::NonRestartingMlirTimer(mlir::Timer &&timer)
    : _running(new bool(false)), _timer(new mlir::Timer(timer)) {}

void NonRestartingMlirTimer::start() {
  if (!(*_running)) {
    _timer->start();
  }
  *_running = true;
}

void NonRestartingMlirTimer::stop() {
  if ((*_running)) {
    _timer->stop();
  }
  *_running = false;
}

mlir::TimingScope NonRestartingMlirTimer::nestAndScope(const char *name) {
  return mlir::TimingScope(_timer->nest(name));
}

} // namespace poptorch_ir
