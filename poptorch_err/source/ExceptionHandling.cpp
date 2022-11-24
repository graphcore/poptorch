// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poptorch_err/ExceptionHandling.hpp"

#include <fstream>
#include <memory>

#include "popart_compiler/Compiler.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#define ERROR_LOG "poptorch_error.log"

namespace poptorch {

ExceptionInfo::~ExceptionInfo() {}

static const int max_log_line_length = 80;

PoptorchErrorInfo convertToPoptorchExceptionOrRethrow(
    const std::exception_ptr &eptr, bool catch_all,
    const std::string &catch_file, uint64_t catch_line) {
  ErrorCategory category = ErrorCategory::Other;
  std::string filename;
  uint64_t line;
  std::string type;
  std::string recovery_action;
  std::string message;
  std::vector<std::string> stack;
  std::string location;

  filename = catch_file;
  line = catch_line;

  try {
    popart_compiler::rethrowPopartOrPoplarException(eptr, catch_file.c_str(),
                                                    catch_line);
    std::rethrow_exception(eptr);
  } catch (const ExceptionInfo &ei) {
    filename = ei.filename();
    line = ei.line();
    category = ei.category();
    type = ei.type();
    message = ei.what();
    for (int i = 0; i < ei.stackDepth(); i++) {
      stack.emplace_back(ei.stack(i));
    }
    recovery_action = ei.recoveryAction();
  } catch (const poptorch::logging::Error &ex) {
    logging::trace("Full error: {}", ex.what());
    message = ex.what();
    type = "poptorch_cpp_error";
    filename = ex.file();
    line = ex.line();
    message = ex.message();
  } catch (const std::out_of_range &ex) {
    message = ex.what();
    type = "std::out_of_range";
  } catch (const std::exception &ex) {
    if (!catch_all) {
      throw;
    }
    message = ex.what();
    type = "std::exception";
  }

  if (std::count(std::begin(message), std::end(message), '\n') >
      max_log_line_length) {
    std::ofstream log;
    log.open(ERROR_LOG);
    log << message;
    log.close();
    message = "See " ERROR_LOG " for details";
  }

  std::stringstream swhat;
  swhat << "In " << filename << ":" << line << ": '" << type
        << "': " << message;
  if (category == ErrorCategory::RuntimeRecoverable) {
    swhat << "\nRecovery action required: " << recovery_action;
  }
  auto ctx = poptorch::logging::LogContext::context();
  if (ctx) {
    location = ctx.get();
    if (!location.empty()) {
      swhat << "\nError raised in:\n" << location;
    }
  }
  PoptorchErrorInfo pe;
  pe.long_message = swhat.str();
  pe.category = category;
  pe.filename = filename;
  pe.line = line;
  pe.type = type;
  pe.recovery_action = recovery_action;
  pe.message = message;
  pe.stack = stack;
  pe.location = location;

  poptorch::logging::LogContext::resetContext();

  return pe;
}

} // namespace poptorch
