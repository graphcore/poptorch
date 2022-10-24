// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef SOURCE_INCLUDE_POPTORCH_TRACEPOINT_HPP
#define SOURCE_INCLUDE_POPTORCH_TRACEPOINT_HPP

#include <algorithm>
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

inline std::string formatPrettyFunction(const char *c) {
  std::string s(c);
  // Find the namespace(s)::class::method substring

  // First locate the start of the arguments
  auto j = std::find(s.begin(), s.end(), '(');

  // Second find the last space before the arguments
  // PRETTY_FUNCTION can return "virtual void poptorch::...."
  auto i = std::find(std::make_reverse_iterator(j), s.rend(), ' ');

  // Get the position of the beginning of the substring
  auto begin_pos = s.size() - static_cast<std::size_t>(i - s.rbegin());
  // Get the size of the substring
  auto size = static_cast<std::size_t>(j - s.begin()) - begin_pos;
  return s.substr(begin_pos, size);
}

#define POPTORCH_TRACEPOINT()                                                  \
  poptorch::logging::Tracepoint tp {                                           \
    poptorch::logging::formatPrettyFunction(__PRETTY_FUNCTION__).c_str()       \
  }

#define POPTORCH_TRACEPOINT_WITH_DEBUG_INFO(debug_info)                        \
  std::stringstream ss;                                                        \
  ss << poptorch::logging::formatPrettyFunction(__PRETTY_FUNCTION__) << " ("   \
     << (debug_info) << ")";                                                   \
  poptorch::logging::Tracepoint tp { ss.str().c_str() }

} // namespace logging
} // namespace poptorch

#endif // SOURCE_INCLUDE_POPTORCH_TRACEPOINT_HPP
