// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"

#include <mlir/Support/Timing.h>

namespace poptorch_ir {

NonRestartingMLIRTimer::NonRestartingMLIRTimer(mlir::Timer &&timer)
    : _running(new bool(false)), _timer(new mlir::Timer(timer)) {}

void NonRestartingMLIRTimer::start() {
  if (!(*_running)) {
    _timer->start();
  }
  *_running = true;
}

void NonRestartingMLIRTimer::stop() {
  if ((*_running)) {
    _timer->stop();
  }
  *_running = false;
}

mlir::TimingScope NonRestartingMLIRTimer::nestAndScope(const char *name) {
  return mlir::TimingScope(_timer->nest(name));
}

} // namespace poptorch_ir
