// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poptorch_err/ExceptionHandling.hpp"

#include <pybind11/pybind11.h>

#include <fstream>
#include <memory>

#include "popart_compiler/Compiler.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#define ERROR_LOG "poptorch_error.log"

namespace py = pybind11;

namespace poptorch {

ExceptionInfo::~ExceptionInfo() {}

namespace {
class Error : public py::object {
public:
  Error() = default;
  Error(handle scope, const char *name, handle base = PyExc_Exception) {
    std::string full_name =
        scope.attr("__name__").cast<std::string>() + std::string(".") + name;
    m_ptr = PyErr_NewException(full_name.c_str(), base.ptr(), nullptr);
    if (hasattr(scope, "__dict__") && scope.attr("__dict__").contains(name)) {
      pybind11::pybind11_fail(
          "Error during initialization: multiple incompatible "
          "definitions with name \"" +
          std::string(name) + "\"");
    }
    scope.attr(name) = *this;
  }

  // Sets the current python myexception to this exception object with the given
  // message
  void setWhat(const std::string &message) { _what = message; }

  const std::string &getWhat() { return _what; }

  void setErrorIndicator() { PyErr_SetString(m_ptr, _what.c_str()); }

  void setMessage(const std::string &message) {
    py::object x = py::cast(message);
    PyObject_SetAttrString(m_ptr, "message", x.ptr());
  }

  void setType(const std::string &type) {
    py::object x = py::cast(type);
    PyObject_SetAttrString(m_ptr, "type", x.ptr());
  }
  void setLocation(const std::string &location) {
    py::object x = py::cast(location);
    PyObject_SetAttrString(m_ptr, "location", x.ptr());
  }

private:
  std::string _what;
};

class RecoverableError : public Error {
public:
  using Error::Error;

  void setRecoveryAction(const std::string &recoveryAction) {
    py::object x = py::cast(recoveryAction);
    PyObject_SetAttrString(m_ptr, "recovery_action", x.ptr());
  }
};
std::unique_ptr<Error> error;
std::unique_ptr<RecoverableError> recoverable_error;
std::unique_ptr<Error> unrecoverable_error;
} // namespace

void initialiseExceptionHandling(pybind11::handle m) {
  error = std::make_unique<Error>(m, "Error");
  recoverable_error =
      std::make_unique<RecoverableError>(m, "RecoverableError", *error);
  unrecoverable_error =
      std::make_unique<Error>(m, "UnrecoverableError", *error);
}

PoptorchError::PoptorchError(const std::string &msg_)
    : torch::PyTorchError(msg_) {}

PyObject *PoptorchError::python_type() { return setupPyError(false); }

void PoptorchError::setErrorIndicator() const { setupPyError(true); }

PyObject *PoptorchError::setupPyError(bool set_indicator) const {
  for (int64_t i = stack.size() - 1; i >= 0; --i) {
    poptorch::logging::LogContext::push(stack[i].c_str());
  }
  Error *err = nullptr;
  switch (category) {
  case ErrorCategory::RuntimeRecoverable: {
    recoverable_error->setRecoveryAction(recovery_action);
    err = recoverable_error.get();
    break;
  }
  case ErrorCategory::RuntimeUnrecoverable: {
    err = unrecoverable_error.get();
    break;
  }
  default: {
    err = error.get();
    break;
  }
  }

  err->setType(type);
  err->setMessage(message);
  err->setLocation(location);
  // Note: on Ubuntu 20.04 PyErr_SetString(), i.e setWhat(),
  // needs to be the last call in register_exception_translator()
  err->setWhat(msg);
  if (set_indicator) {
    err->setErrorIndicator();
  }
  return err->ptr();
}

static const int max_log_line_length = 80;

void rethrowPoptorchException(const std::exception_ptr &eptr,
                              const std::string &catch_file,
                              uint64_t catch_line) {
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
    rethrowPopartOrPoplarException(eptr, catch_file.c_str(), catch_line);
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
  PoptorchError pe(swhat.str());
  pe.category = category;
  pe.filename = filename;
  pe.line = line;
  pe.type = type;
  pe.recovery_action = recovery_action;
  pe.message = message;
  pe.stack = stack;
  pe.location = location;

  poptorch::logging::LogContext::resetContext();

  throw pe;
}

} // namespace poptorch
