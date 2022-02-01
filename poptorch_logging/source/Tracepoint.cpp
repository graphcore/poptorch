// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poptorch_logging/Tracepoint.hpp"

#include <pvti/pvti.hpp>

#include "poptorch_logging/Error.hpp"

namespace poptorch {

namespace logging {

namespace detail {

class TracepointImpl : public pvti::Tracepoint {
public:
  explicit TracepointImpl(const char *label_)
      : pvti::Tracepoint(&TracepointImpl::channel, label_), ctx(label_) {}
  ~TracepointImpl() = default;
  static pvti::TraceChannel channel;
  LogContext ctx;
};

pvti::TraceChannel TracepointImpl::channel = {"poptorch"};
} // namespace detail

Tracepoint::Tracepoint(const char *label)
    : _impl(std::make_unique<detail::TracepointImpl>(label)) {}

void Tracepoint::begin(const char *label) {
  pvti::Tracepoint::begin(&detail::TracepointImpl::channel, label);
}

void Tracepoint::end(const char *label) {
  pvti::Tracepoint::end(&detail::TracepointImpl::channel, label);
}

Tracepoint::~Tracepoint() = default;

} // namespace logging
} // namespace poptorch
