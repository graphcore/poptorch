// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef SOURCE_INCLUDE_POPTORCH_TRACEPOINT_HPP
#define SOURCE_INCLUDE_POPTORCH_TRACEPOINT_HPP

#include <memory>
#include <string>

namespace poptorch {
namespace logging {

namespace detail {
class TracepointImpl;
}

/** RAII class to create tracepoints
 */
class Tracepoint {
public:
  explicit Tracepoint(const char *label);
  ~Tracepoint();

  static void begin(const char *label);
  static void end(const char *label);

private:
  std::unique_ptr<detail::TracepointImpl> _impl;
};

} // namespace logging
} // namespace poptorch

#endif // SOURCE_INCLUDE_POPTORCH_TRACEPOINT_HPP
