// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

namespace logging {

error::error(const char *s) : std::runtime_error(s) { logging::err(what()); }

} // namespace logging
