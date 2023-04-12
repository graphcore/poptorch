// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <ATen/ATen.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "poptorch_err/ExceptionHandling.hpp"
#include "poptorch_err/ExceptionInfo.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/LoggingLight.hpp"
#include "poptorch_logging/Tracepoint.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/LowerToPopart.hpp"
#include "poptorch/LowerToPopartFactories.hpp"
#include "poptorch/SessionOptionsParser.hpp"
#include "poptorch/Utils.hpp"

#include "popart_compiler/CodeletsCompilation.hpp"
#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/Utils.hpp"

#include "pytorch_bridge/CompilerOptions.hpp"

// All the functions here are called directly from python, therefore it's ok for
// us to catch all exceptions and convert them to PoptorchError
#define PTC(f)                                                                 \
  PoptorchCatchWrapperImpl<poptorch::throwPoptorchError, /*catch_all=*/true,   \
                           decltype(&(f)), f>::wrap

namespace poptorch {
namespace {

// Everything in this namespace is a workaround because
// torch::jit::toTraceableStack() is broken:torch::jit::as_module() fails to
// initialise its static local variable ScriptModule and segfaults as a
// result.
namespace jit {

using namespace torch::jit;

TypePtr inferType(py::handle input) {

  // Try tensor types
  if (THPVariable_Check(input.ptr())) {
    return TensorType::get();
  }

  if (input.is(py::none())) {
    return NoneType::get();
  }

  if (six::isTuple(input)) {
    py::tuple tuple = py::cast<py::tuple>(input);
    std::vector<TypePtr> element_types;
    element_types.reserve(tuple.size());

    for (py::handle elem : tuple) {
      element_types.push_back(inferType(elem));
    }
    return TupleType::create(element_types);
  } else if (PyDict_Check(input.ptr())) {
    // Check to make sure we can generate useful input/output types
    auto dict = py::cast<py::dict>(input);
    size_t len = py::len(dict);
    ERROR_ON_MSG(len == 0, "Dictionary inputs must have entries");
    TypePtr key_type = nullptr;
    TypePtr value_type = nullptr;

    for (auto entry : dict) {
      // Try to infer the key type and unify it with the existing one
      auto entry_key_type = inferType(entry.first);
      auto unified_key = unifyOrInitializeType(key_type, entry_key_type);
      ERROR_ON_MSG(!unified_key,
                   c10::str("Dictionary inputs to traced functions must have "
                            "consistent type. Found ",
                            key_type->repr_str(), " and ",
                            entry_key_type->repr_str()));

      // Try to infer the value type and unify it with the existing one
      auto entry_value_type = inferType(entry.second);
      auto unified_value = unifyOrInitializeType(value_type, entry_value_type);
      ERROR_ON_MSG(!unified_value,
                   c10::str("Dictionary inputs to traced functions must have "
                            "consistent type. Found ",
                            value_type->repr_str(), " and ",
                            entry_value_type->repr_str()));

      key_type = *unified_key;
      value_type = *unified_value;
    }
    return DictType::create(key_type, value_type);
  } else if (PyList_Check(input.ptr())) {
    auto list = py::cast<py::list>(input);
    size_t len = py::len(list);
    ERROR_ON_MSG(len == 0, "List trace inputs must have elements");

    TypePtr element_type = nullptr;
    for (auto elem : list) {
      auto this_element_type = inferType(elem);
      auto unified_type =
          unifyOrInitializeType(element_type, this_element_type);
      ERROR_ON_MSG(!unified_type,
                   c10::str("List inputs to traced functions must have "
                            "consistent element type. Found ",
                            element_type->repr_str(), " and ",
                            this_element_type->repr_str()));
      element_type = *unified_type;
    }
    return ListType::create(element_type);
  }
  ERROR("Only nested lists and tuples of tensors are supported");
}

// Cut down version of torch::jit::toTraceableStack which only supports nested
// tuples and lists of tensors.
Stack toTraceableStack(const py::tuple &inputs) {
  return toIValue(inputs, inferType(inputs)).toTupleRef().elements().vec();
}

} // namespace jit

template <typename Func> class CallOnExit : Func {
public:
  explicit CallOnExit(Func f) : Func(std::move(f)) {}
  ~CallOnExit() { std::invoke(*static_cast<Func *>(this)); }
};

// Keep a static map to gather up all the cpu calls.
CPUCallbackMap callbacks;

bool alreadyRegistered(const std::string &ID) {
  return callbacks.find(ID) != callbacks.end();
}

void registerBuffersWithCallback(
    const std::string &ID,
    std::vector<at::Tensor> &input_tensors, // NOLINT
    std::vector<at::Tensor> &output_tensors // NOLINT
) {
  auto itr = callbacks.find(ID);

  ERROR_ON_MSG(itr == callbacks.end(), "Callback has not been registered.");

  popart_compiler::CallbackMetadata &metadata = itr->second;

  // Track the input tensors. Our python creates a persistent storage location
  // for the inputs and outputs.
  for (at::Tensor &tensor : input_tensors) {
    metadata.input_pointers.push_back(tensor.data_ptr());
  }

  // Same for output.
  for (at::Tensor &tensor : output_tensors) {
    tensor = tensor.contiguous();
    metadata.output_pointers.push_back(tensor.data_ptr());
  }
}

// Python interface to map a given CPU op with the IR calls.
void registerCPUCallBack(const py::object &obj, const std::string &ID) {
  // Map the string identifier to the metadata.
  bool inserted;
  decltype(callbacks)::iterator it;
  std::tie(it, inserted) = callbacks.try_emplace(ID);

  // Skip if we've already added a callback for this function.
  if (!inserted) {
    return;
  }

  // Structure to store the information given by python to be forwarded to the
  // backend.
  popart_compiler::CallbackMetadata &metadata = it->second;

  // Wrap that in a lambda so we don't have to expose the naked pytorch function
  // pointer thing.
  metadata.the_callback = [=]() {
    // We wrap the user call in a function called "execute"
    obj.attr("execute")();
  };

  metadata.buffer_registration_callback = [=]() {
    obj.attr("registerPersistentData")();
  };
}

void initCallbackBuffers() {
  for (auto &pair : callbacks) {
    pair.second.buffer_registration_callback();
  }
}

class PybindValue : public IPyValue {
public:
  template <typename T,
            std::enable_if_t<std::is_base_of<py::object, T>::value, int> = 0>
  explicit PybindValue(T obj) {
    _maybe_obj = obj;
    _value = _maybe_obj;
  }

  template <typename T,
            std::enable_if_t<!std::is_base_of<py::object, T>::value, int> = 0>
  explicit PybindValue(T handle) : _value(handle) {}

  std::function<void(int, int)> toFunction() const override {
    py::function py_func = _value.cast<py::function>();
    return [py_func](int x, int y) {
      py::gil_scoped_acquire acquire;
      py_func(x, y);
    };
  }

  bool isBoolean() const override { return py::isinstance<py::bool_>(_value); }

  bool toBoolean() const override { return _value.cast<bool>(); }

  bool isDouble() const override {
    // Python's float type is actually double
    // precision.
    return py::isinstance<py::float_>(_value);
  }

  double toDouble() const override { return _value.cast<double>(); }

  bool isInt() const override { return py::isinstance<py::int_>(_value); }

  std::uint64_t toUInt64() const override {
    return _value.cast<std::uint64_t>();
  }

  std::int64_t toInt64() const override { return _value.cast<std::int64_t>(); }

  bool isString() const override { return py::isinstance<py::str>(_value); }

  std::string toString() const override {
    if (isString()) {
      return _value.cast<std::string>();
    }
    if (isInt()) {
      return std::to_string(_value.cast<std::uint64_t>());
    }
    ERROR("Don't know how to convert type " << _value.get_type()
                                            << " to string");
  }

  bool isSetListOrTuple() const override {
    return py::isinstance<py::set>(_value) ||
           py::isinstance<py::list>(_value) ||
           py::isinstance<py::tuple>(_value);
  }

  void forEachInList(std::function<void(const IPyValue &)> fn) const override {
    for (auto option : _value.cast<py::list>()) {
      fn(PybindValue(option));
    }
  }

  bool isDict() const override { return py::isinstance<py::dict>(_value); }

  void forEachInDict(std::function<void(const IPyValue &, const IPyValue &)> fn)
      const override {
    for (auto option : _value.cast<py::dict>()) {
      fn(PybindValue(option.first), PybindValue(option.second));
    }
  }

  std::unique_ptr<IPyValue> getFromDict(const std::string &key) const override {
    auto dict = _value.cast<py::dict>();
    if (!dict.contains(key)) {
      return nullptr;
    }
    return std::make_unique<PybindValue>(dict[key.c_str()]);
  }
  std::uint64_t getListSize() const override {
    return _value.cast<py::list>().size();
  }
  std::unique_ptr<IPyValue>
  getFromList(const std::uint64_t index) const override {
    auto list = _value.cast<py::list>();
    if (index >= list.size()) {
      return nullptr;
    }
    return std::make_unique<PybindValue>(list[index]);
  }

  std::string type() const override { return py::str(_value.get_type()); }

private:
  // pybind11 handles do not keep a reference to the python object so it might
  // disappear if the parent doesn't hold a reference to it, so just to be safe
  // keep a reference if possible.
  py::object _maybe_obj;
  py::handle _value;
};

template <typename T>
T getOptimizerValue(const py::dict &d, const std::string &key) {
  ERROR_ON_MSG(!d.contains(key), "Missing optimizer value for '"
                                     << key << "' in "
                                     << py::str(d.cast<py::object>()));
  return d[key.c_str()].cast<T>();
}

template <typename T>
void getOptimizerValue(T &value, const py::dict &d, const std::string &key) {
  value = getOptimizerValue<T>(d, key);
}

void copyParametersDict(popart_compiler::Optimizer *out, const py::dict &in) {
  logging::LogContext ctx_func("copyParametersDict");
  out->parameters.resize(in.size());
  std::uint64_t param_idx = 0;
  for (auto optimizer_field : in) {
    auto &param = out->parameters[param_idx];
    param_idx++;

    const std::string name = optimizer_field.first.cast<std::string>();
    logging::LogContext ctx("attr: " + name);
    std::pair<float, bool> p =
        optimizer_field.second.cast<std::pair<float, bool>>();

    ERROR_ON(name.size() >= sizeof(param.name));
    // We need to use a C-style string here to avoid ABI issues.
    snprintf(reinterpret_cast<char *>(param.name), sizeof(param.name), "%s",
             name.c_str());
    param.value = p.first;
    param.is_const = p.second;
  }
}

// Process the user provided dictionary and extract the relevant optimizer
// information.
std::vector<popart_compiler::Optimizer> parseOptimizers(const py::dict &opt) {
  if (opt.empty()) {
    return {};
  }

  popart_compiler::OptimizerType type = popart_compiler::OptimizerType::NONE;
  std::uint64_t num_groups;
  type = static_cast<popart_compiler::OptimizerType>(
      getOptimizerValue<std::uint64_t>(opt, "optimizer_type"));
  auto defaults = getOptimizerValue<py::dict>(opt, "defaults");
  auto groups = getOptimizerValue<py::list>(opt, "groups");
  num_groups = groups.size();
  std::vector<popart_compiler::Optimizer> optimizers;
  // Note: all the group variables and optimizer variables are
  // automatically forwarded to the Compiler backend however
  // the optimizer attributes are extracted here.
  bool use_tf_variant = false;
  if (type == popart_compiler::OptimizerType::RMSPROP ||
      type == popart_compiler::OptimizerType::RMSPROP_CENTERED) {
    getOptimizerValue(use_tf_variant, opt, "useTfVariant");
  }

  float max_grad_norm = std::numeric_limits<float>::infinity();
  if (opt.contains("maxGradNorm")) {
    getOptimizerValue(max_grad_norm, opt, "maxGradNorm");
  }

  if (opt.contains("accumType")) {
    bool accum_type = false;
    bool first_order_momentum_accum_type = false;
    bool second_order_momentum_accum_type = false;

    // Indicate whether the optimizer should use float16 types
    getOptimizerValue(accum_type, opt, "accumType");

    if (type == popart_compiler::OptimizerType::SGD1 ||
        type == popart_compiler::OptimizerType::SGD2) {
      getOptimizerValue(first_order_momentum_accum_type, opt,
                        "velocityAccumType");
    } else {
      getOptimizerValue(first_order_momentum_accum_type, opt,
                        "firstOrderMomentumAccumType");
      getOptimizerValue(second_order_momentum_accum_type, opt,
                        "secondOrderMomentumAccumType");
    }
    // Create one Optimizer per parameter group + 1 for defaults
    for (std::uint64_t i = 0; i <= num_groups; ++i) {
      optimizers.emplace_back(type, accum_type, first_order_momentum_accum_type,
                              second_order_momentum_accum_type, use_tf_variant,
                              max_grad_norm);
    }
  } else {
    // Create one Optimizer per parameter group + 1 for defaults
    for (std::uint64_t i = 0; i <= num_groups; ++i) {
      optimizers.emplace_back(type, use_tf_variant, max_grad_norm);
    }
  }

  copyParametersDict(optimizers.data(), defaults);
  // For each group copy all the attributes
  // Start at 1: index 0 is 'defaults'
  std::uint64_t group = 1;
  for (auto group_attr : groups) {
    copyParametersDict(&optimizers[group], group_attr.cast<py::dict>());
    ++group;
  }
  return optimizers;
}

std::map<std::string, void *>
getParameterBuffers(const pybind11::tuple &names,
                    const pybind11::tuple &tensors) {
  ERROR_ON(names.size() != tensors.size());
  std::map<std::string, void *> parameters;
  torch::jit::Stack stack = jit::toTraceableStack(tensors);
  for (std::uint64_t i = 0; i < names.size(); ++i) {
    parameters.insert(
        {names[i].cast<std::string>(), stack[i].toTensor().data_ptr()});
  }
  return parameters;
}

// We have three sets of tensors.
// 1. Tensors in the graph from jit::trace.
// 2. Tensors in the original user model.
// 3. Tensors in the graph from jit::trace which lowerGraph has removed unused
// tensors from. We remap them by mapping the indices of 1. to the tensors of 3.
// and then creating a new vector using 3 with that map as a guide to tell us
// which tensors have been culled.
std::vector<at::Tensor>
remapTensors(const pybind11::dict &python_tensors,
             const pybind11::dict &model_parameters,
             const std::vector<at::Tensor> &traced_tensors) {
  // Create a set of the pointers actually in use.
  std::unordered_map<void *, std::size_t> tensor_pointers;

  for (std::size_t i = 0; i < traced_tensors.size(); ++i) {
    tensor_pointers.insert({traced_tensors[i].data_ptr(), i});
  }

  std::vector<at::Tensor> returnee;
  returnee.resize(traced_tensors.size());

  for (auto element : model_parameters) {
    auto option_name = element.first.cast<std::string>();

    // Get the original tensor which the.
    auto dict_itr = python_tensors[element.first];
    at::Tensor traced_tensor = dict_itr.cast<at::Tensor>();

    auto itr = tensor_pointers.find(traced_tensor.data_ptr());
    if (itr != tensor_pointers.end()) {
      at::Tensor tensor = element.second.cast<at::Tensor>();
      returnee[itr->second] = tensor;
    }
  }

  return returnee;
}

// python_names and python_tensors are the parameters from the python trace.
// And trace_tensors is a subset of python_tensors (The unused parameters have
// been removed). So we build a map[tensor] = name based on the python trace
// which we then use to build the list of the names of the parameters in
// traced_tensors.
std::vector<std::string>
getParameterNames(const pybind11::dict &python_tensors,
                  const std::vector<at::Tensor> &traced_tensors) {
  // Create a set of the pointers actually in use.
  std::unordered_map<void *, std::size_t> tensor_pointers;

  for (std::size_t i = 0; i < traced_tensors.size(); ++i) {
    tensor_pointers.insert({traced_tensors[i].data_ptr(), i});
  }

  // Get the names of each tensor which hasn't been removed as unused.
  std::vector<std::string> names;
  names.resize(tensor_pointers.size());

  // Extract the python strings into an actual language.
  for (auto element : python_tensors) {
    at::Tensor tensor = element.second.cast<at::Tensor>();

    auto itr = tensor_pointers.find(tensor.data_ptr());

    if (itr != tensor_pointers.end()) {
      std::string option_name = element.first.cast<std::string>();
      names[itr->second] = option_name;
    }
  }

  return names;
}

AnchorList parseAnchors(const py::list &list) {
  AnchorList map;
  for (auto elem : list) {
    auto anchor = elem.cast<py::list>();
    map.push_back(Anchor(anchor[0].cast<std::string>(),
                         anchor[2].cast<std::uint64_t>(),
                         anchor[3].cast<std::uint64_t>()));
  }
  return map;
}

void parseSessionOptionsVoid(const py::dict &opts) {
  SessionOptionsParser{PybindValue(opts)};
}

void buildTensorList(const torch::jit::IValue &value,
                     std::vector<at::Tensor> *tensors,
                     bool allow_tensor_only = false) {
  if (value.isTuple()) {
    ERROR_ON(allow_tensor_only);
    for (auto &element : value.toTuple()->elements()) {
      buildTensorList(element, tensors);
    }
  } else if (value.isList()) {
    ERROR_ON(allow_tensor_only);
    for (const auto element : value.toList()) {
      buildTensorList(element, tensors);
    }
  } else if (value.isTensor()) {
    tensors->push_back(value.toTensor());
  } else {
    ERROR("Unsupported value " << value.tagKind());
  }
}

torch::jit::script::Module *asModule(py::handle h) {
  return reinterpret_cast<torch::jit::script::Module *>(
      pybind11::detail::values_and_holders(
          reinterpret_cast<pybind11::detail::instance *>(h.ptr()))
          .begin()
          ->value_ptr());
}

void identifyZeroSizedTensors(const std::vector<at::Tensor> &tensors) {
  for (const at::Tensor &tensor : tensors) {
    auto sizes = tensor.sizes();
    if (std::any_of(sizes.begin(), sizes.end(),
                    [](auto dim) { return dim == 0; })) {
      std::stringstream err;
      err << "Zero-sized tensors are unsupported (Got shape [";
      for (std::size_t i = 0; i < sizes.size() - 1; i++) {
        err << sizes[i] << ", ";
      }
      err << sizes[sizes.size() - 1] << "]).";
      ERROR(err.str());
    }
  }
}

poptorch::LowerToPopart
lowerToPopartFromDispatch(const pybind11::dict &options,
                          const py::function &attribute_accessor, bool training,
                          const py::dict &opt_dict, const py::list &anchors) {
  auto cleanup = CallOnExit([] {
    // Clear the callbacks after compilation.
    callbacks.clear();
  });

  SessionOptionsParser options_parser{PybindValue(options)};

  AnchorList anchors_list = parseAnchors(anchors);
  std::vector<popart_compiler::Optimizer> optimizers =
      parseOptimizers(opt_dict);

  return lowerToPopartFromDispatch(
      options_parser, training, std::move(anchors_list),
      []() { initCallbackBuffers(); }, std::move(optimizers),
      [&attribute_accessor](const std::string &attributes_id_str) {
        return std::make_unique<PybindValue>(
            attribute_accessor(attributes_id_str));
      },
      callbacks);
}

void mapParamsToNames(const pybind11::tuple &names,
                      const pybind11::tuple &tensors) {
  ERROR_ON(names.size() != tensors.size());
  torch::jit::Stack stack = jit::toTraceableStack(tensors);
  for (uint64_t i = 0; i < names.size(); ++i) {
    const auto name = names[i].cast<std::string>();
    const auto tensor = stack[i].toTensor();
    setParameterName(tensor, name);
  }
}

void setPerReplica(const std::string &param_name, py::handle tensor,
                   int comm_group_type, int shards,
                   int variable_retrieval_mode) {
  at::Tensor t = torch::jit::toTypeInferredIValue(tensor).toTensor();
  setParameterPerReplica(param_name, t, comm_group_type, shards,
                         variable_retrieval_mode);
}

std::string convertToString(const std::vector<char> &str) {
  return std::string(str.data(), str.size());
}

std::vector<char> convertToCharVec(const std::string &str) {
  return std::vector<char>(str.begin(), str.end());
}

pybind11::list toPythonList(std::vector<at::Tensor> &&outputs) {
  pybind11::list pylist(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    pylist[i] = torch::jit::toPyObject(std::move(outputs[i]));
  }
  return pylist;
}

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

/*
 * This structure enables poptorch.Error objects to be thrown python-side from
 * both our pybind11 interface and torch's own. Our pybind11 exception handler
 * catches this class specifically, whilst torch's catches any PyTorchError
 * subclass and uses it to deduce the python type using the overridden
 * python_type() method.
 * The function convertToPoptorchExceptionOrRethrow() processes all the
 * exception types we're interested in, extracts detail, and marshals them as
 * instances of PoptorchErrorInfo which is then used to create instances of this
 * class. We put try..catch wrappers round every pybind11 entry point using the
 * macro CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION and pass them to
 * convertToPoptorchExceptionOrRethrow().
 */
struct PoptorchError : public torch::PyTorchError {
public:
  explicit PoptorchError(const PoptorchErrorInfo &info)
      : torch::PyTorchError(info.long_message), _info(info) {}
  PyObject *python_type() override { return setupPyError(false); }
  void setErrorIndicator() const { setupPyError(true); }

private:
  PyObject *setupPyError(bool set_indicator) const;

public:
  const PoptorchErrorInfo _info;
};

PyObject *PoptorchError::setupPyError(bool set_indicator) const {
  for (int64_t i = _info.stack.size() - 1; i >= 0; --i) {
    poptorch::logging::LogContext::push(_info.stack[i].c_str());
  }
  Error *err = nullptr;
  switch (_info.category) {
  case ErrorCategory::RuntimeRecoverable: {
    recoverable_error->setRecoveryAction(_info.recovery_action);
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

  err->setType(_info.type);
  err->setMessage(_info.message);
  err->setLocation(_info.location);
  // Note: on Ubuntu 20.04 PyErr_SetString(), i.e setWhat(),
  // needs to be the last call in register_exception_translator()
  err->setWhat(_info.long_message);
  if (set_indicator) {
    err->setErrorIndicator();
  }
  return err->ptr();
}

void doThrowPoptorchError(const PoptorchErrorInfo &info) {
  throw PoptorchError(info);
}
} // namespace

namespace bindings {

void initialiseExceptionHandling(pybind11::handle m) {
  error = std::make_unique<Error>(m, "Error");
  recoverable_error =
      std::make_unique<RecoverableError>(m, "RecoverableError", *error);
  unrecoverable_error =
      std::make_unique<Error>(m, "UnrecoverableError", *error);
  poptorch::setPoptorchErrorThrower(doThrowPoptorchError);
}

void copyWeightsToHostImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors) {
  poptorch::logging::Tracepoint tp{"copyWeightsToHost"};
  // Copy the weights or warn if this is before first time compilation.
  if (!executable) {
    logging::log(
        logging::Level::Warn,
        "Call to copyWeightsToHost ignored as model has not been compiled "
        "(PopTorch will compile models on first invocation).");
  } else {
    executable->copyWeightsToHost(
        getParameterBuffers(parameter_names, parameter_tensors));
  }
}

void copyWeightsToDeviceImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors) {
  poptorch::logging::Tracepoint tp{"copyWeightsToDevice"};
  // Copy the weights or warn if this is before first time compilation.
  if (!executable) {
    logging::log(
        logging::Level::Warn,
        "Call to copyWeightsToDevice ignored as model has not been compiled "
        "(PopTorch will compile models on first invocation).");
  } else {
    executable->copyWeightsToDevice(
        getParameterBuffers(parameter_names, parameter_tensors));
  }
}

void copyNamedBuffersToDeviceImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &buffer_names,
    const pybind11::tuple &buffer_tensors) {
  poptorch::logging::Tracepoint tp{"copyNamedBuffersToDevice"};
  // Copy the named buffers or warn if this is before first time compilation.
  if (!executable) {
    logging::log(
        logging::Level::Warn,
        "Call to copyNamedBuffersToDevice ignored as model has not been "
        "compiled (PopTorch will compile models on first invocation).");
  } else {
    executable->copyNamedBuffersToDevice(
        getParameterBuffers(buffer_names, buffer_tensors));
  }
}

std::string
getPopartIR(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  ERROR_ON_MSG(!executable, "No built executable");
  return executable->getPopartIR();
}

py::set
getTensorNames(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  ERROR_ON_MSG(!executable, "No built executable");
  return py::cast(executable->getTensorNames());
}

void detachFromDevice(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  executable->detachFromDevice();
}

void attachToDevice(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  executable->attachToDevice();
}

bool isAttachedToDevice(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  ERROR_ON_MSG(!executable, "No built executable");
  return executable->isAttachedToDevice();
}

void setLogLevel(std::uint64_t level) {
  ERROR_ON(level > static_cast<std::uint64_t>(logging::Level::Off) ||
           level == 5);
  logging::setLogLevel(static_cast<logging::Level>(level));
}

void loadEngineAndConnectStreams(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  executable->loadEngineAndConnectStreams();
}

void updateOptimizers(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const py::dict &optimizer_dict) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  // Create an empty optimizer for inference, this will not be applied.
  std::vector<popart_compiler::Optimizer> optimizers =
      parseOptimizers(optimizer_dict);

  executable->updateOptimizers(optimizers);
}

std::vector<pybind11::object>
execute(const std::shared_ptr<poptorch::PoplarExecutable> &executable,
        const pybind11::tuple &inputs) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack input_stack = jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> input_tensors;
  for (const torch::jit::IValue &value : input_stack) {
    buildTensorList(value, &input_tensors);
  }

  std::vector<at::IValue> output_tensors = executable->run(input_tensors);

  std::vector<pybind11::object> returnee;

  // Reshape the output tensors in the structure expected by the user
  auto tensor_it = output_tensors.begin();
  const auto &output_types = executable->outputTypes();
  auto type_it = output_types.begin();
  ERROR_ON(type_it == output_types.end());

  // First tuple encodes the number of (actual) outputs
  std::uint64_t num_outputs = type_it->num_elements;
  std::function<pybind11::object()> process_output;
  process_output = [&]() -> pybind11::object { // NOLINT
    ERROR_ON_MSG(type_it == output_types.end(), "Invalid OutputTypes object");
    switch (type_it->type) {
    case popart_compiler::OutputElemType::Tensor: {
      ERROR_ON_MSG(tensor_it == output_tensors.end(),
                   "Not enough tensors to unpack");
      auto object = torch::jit::toPyObject(*tensor_it);
      tensor_it++;
      return object;
    }
    case popart_compiler::OutputElemType::Tuple: {
      std::int64_t num_elements = type_it->num_elements;
      pybind11::tuple pytuple(num_elements);
      for (std::int64_t i = 0; i < num_elements; ++i) {
        type_it++;
        pytuple[i] = process_output();
      }
      return std::move(pytuple);
    }
    case popart_compiler::OutputElemType::List: {
      std::int64_t num_elements = type_it->num_elements;
      pybind11::list pylist(num_elements);
      for (std::int64_t i = 0; i < num_elements; ++i) {
        type_it++;
        pylist[i] = process_output();
      }
      return std::move(pylist);
    }
    default:
      ERROR("Unsupported OutputType");
    }
  };

  for (std::uint64_t i = 0; i < num_outputs; ++i) {
    type_it++;
    returnee.push_back(process_output());
  }
  ERROR_ON_MSG(tensor_it != output_tensors.end(),
               "Not all the output tensors were unpacked");

  return returnee;
}

void setRngState(std::shared_ptr<poptorch::PoplarExecutable> &executable,
                 std::uint64_t seed,
                 const std::vector<std::uint32_t> &rng_state) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  auto &compiler = executable->getCompiler();
  compiler.setRngState(seed, rng_state);
}

std::uint64_t
getRandomSeed(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  const auto &compiler = executable->getCompiler();
  return compiler.getRandomSeed();
}

std::vector<std::uint32_t>
getRngState(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  const auto &compiler = executable->getCompiler();
  return compiler.getRngState();
}

py::dict readOptimizerState(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  py::dict optim_state;
  py::dict state_tensors;
  py::dict param_tensors;
  ERROR_ON_MSG(!executable, "No built executable");
  auto &compiler = executable->getCompiler();
  std::vector<popart_compiler::TensorMetadata> metadata_list =
      compiler.optimizerTensorMetadataList();

  std::vector<void *> host_buffers;
  for (const popart_compiler::TensorMetadata &meta : metadata_list) {
    at::Tensor tensor =
        at::empty({meta.shape}, onnxStrToScalarType(meta.dtype)).contiguous();

    if (meta.num_bytes == -1) {
      // num_bytes == -1 indicates it's an optimiser state tensor (variable)
      host_buffers.push_back(tensor.data_ptr());
      state_tensors[py::cast(meta.id)] = py::cast(tensor);
    } else {
      // Otherwise it's a stream/constant optimiser parameter that we can copy
      // immediately
      std::memcpy(tensor.data_ptr(), meta.data, meta.num_bytes);
      param_tensors[py::cast(meta.id)] = py::cast(tensor);
    }
  }
  compiler.fillHostOptimizerStateTensorData(host_buffers);
  optim_state["ipu_state"] = std::move(state_tensors);
  optim_state["ipu_param"] = std::move(param_tensors);
  return optim_state;
}

void writeOptimizerState(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const py::dict &optim_state) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  auto &compiler = executable->getCompiler();
  std::vector<popart_compiler::TensorMetadata> metadata_list =
      compiler.optimizerTensorMetadataList();

  std::vector<void *> host_buffers;
  auto state = optim_state["ipu_state"];
  auto params = optim_state["ipu_param"];

  for (const popart_compiler::TensorMetadata &meta : metadata_list) {
    if (meta.num_bytes == -1) {
      // num_bytes == -1 indicates it's an optimiser state tensor (variable)
      if (!state.contains(py::cast(meta.id))) {
        logging::log(
            logging::Level::Warn,
            std::string("writeOptimizerState: ignoring missing state " +
                        std::string(meta.id))
                .c_str());
        host_buffers.push_back(nullptr);
        continue;
      }
      at::Tensor tensor = state[py::cast(meta.id)].cast<at::Tensor>();
      host_buffers.push_back(tensor.data_ptr());
    } else {
      if (!params.contains(py::cast(meta.id))) {
        logging::log(
            logging::Level::Warn,
            std::string("writeOptimizerState: ignoring missing parameter " +
                        std::string(meta.id))
                .c_str());
        continue;
      }
      // Otherwise it's a stream/constant optimiser parameter that we can copy
      // immediately
      at::Tensor tensor = params[py::cast(meta.id)].cast<at::Tensor>();
      std::memcpy(meta.data, tensor.data_ptr(), meta.num_bytes);
    }
  }
  compiler.writeDeviceOptimizerStateTensorData(host_buffers);
}

std::vector<pybind11::object>
getTimestamps(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  const auto &compiler = executable->getCompiler();
  popart_compiler::Timestamps ts = compiler.getTimestamps();

  py::list input;
  py::list input_complete;
  py::list output;
  py::list output_complete;

  for (const auto &t : ts.input) {
    input.append(py::cast(t));
  }
  for (const auto &t : ts.input_complete) {
    input_complete.append(py::cast(t));
  }
  for (const auto &t : ts.output) {
    output.append(py::cast(t));
  }
  for (const auto &t : ts.output_complete) {
    output_complete.append(py::cast(t));
  }

  return {input, input_complete, output, output_complete};
}

bool pyIsGraphNondeterministic(py::handle h) {
  auto *module = asModule(h);
  auto forward = module->get_method("forward");
  auto graph_and_tensors =
      torch::jit::LowerGraph(*forward.graph(), module->_ivalue());
  auto graph = graph_and_tensors.first;
  const auto &nodes = graph->nodes();
  return std::any_of(nodes.begin(), nodes.end(), [](const torch::jit::Node *n) {
    return poptorch::isNondeterministic(*n);
  });
}

void saveExecutableToFile(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const std::string &export_filename) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  executable->getCompiler().saveExecutableToFile(export_filename.c_str());
}

void appendPoptorchMetadataToFile(const std::string &serialized_poptorch_data,
                                  const std::string &export_filename) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  popart_compiler::Compiler::appendPoptorchMetadataToFile(
      serialized_poptorch_data.c_str(), serialized_poptorch_data.size(),
      export_filename.c_str());
}

uint64_t
cycleCount(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  ERROR_ON_MSG(!executable, "No built executable");
  return executable->getCompiler().getCycleCount();
}

py::bytes importPoptorchMetadataFromFile(const std::string &import_filename) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  std::vector<char> metadata_buffer =
      popart_compiler::Compiler::importPoptorchMetadataFromFile(
          import_filename.c_str());
  return py::bytes(metadata_buffer.data(), metadata_buffer.size());
}

std::shared_ptr<poptorch::PoplarExecutable> processDispatchAndImportExecutable(
    const pybind11::dict &options, const py::function &attribute_accessor,
    bool is_training, const py::dict &opt_dict, const py::list &anchors,
    const std::string &import_filename) {
  auto lower = lowerToPopartFromDispatch(options, attribute_accessor,
                                         is_training, opt_dict, anchors);
  return lower.loadExecutableFromFile(import_filename);
}
std::shared_ptr<poptorch::PoplarExecutable> compileWithManualTracing(
    const pybind11::dict &options, const py::function &attribute_accessor,
    bool is_training, const py::dict &opt_dict, const py::list &anchors) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  logging::log(logging::Level::Debug, "Compile with manual tracing");
  auto lower = lowerToPopartFromDispatch(options, attribute_accessor,
                                         is_training, opt_dict, anchors);
  py::gil_scoped_release release;
  return lower.compile();
}

void setPopartLogLevelUInt(std::uint64_t level) {
  ERROR_ON(level > static_cast<std::uint64_t>(logging::Level::Off) ||
           level == 5);
  popart_compiler::setPopartLogLevel(static_cast<logging::Level>(level));
}

} // namespace bindings

} // namespace poptorch

PYBIND11_MODULE(poptorch_core, m) { // NOLINT
  py::class_<poptorch::PoplarExecutable,
             std::shared_ptr<poptorch::PoplarExecutable>>
      give_me_a_name(m, "InternalPoplarExecutable");
  py::class_<poptorch::CompilerOptions>(m, "CompilerOptions")
      .def(py::init<>())
      .def_property(
          "source_location_excludes",
          [](const poptorch::CompilerOptions &options) {
            std::vector<std::string> excludes;
            std::transform(options.dispatcher.source_location_excludes.begin(),
                           options.dispatcher.source_location_excludes.end(),
                           std::back_inserter(excludes),
                           &poptorch::convertToString);
            return excludes;
          },
          [](poptorch::CompilerOptions &options,
             const std::vector<std::string> &val) {
            options.dispatcher.source_location_excludes.clear();
            std::transform(
                val.begin(), val.end(),
                std::back_inserter(options.dispatcher.source_location_excludes),
                &poptorch::convertToCharVec);
          },
          "When printing the IR all the frames containing one of the excluded"
          "strings will be ignored.\n\n"
          "This is helpful to get the IR to trace back to user code rather"
          "than some function inside a framework.");

  m.def("isGraphNondeterministic",
        PTC(poptorch::bindings::pyIsGraphNondeterministic));
  m.def("saveExecutableToFile", PTC(poptorch::bindings::saveExecutableToFile));
  m.def("appendPoptorchMetadataToFile",
        PTC(poptorch::bindings::appendPoptorchMetadataToFile));
  m.def("cycleCount", PTC(poptorch::bindings::cycleCount));
  m.def("importPoptorchMetadataFromFile",
        PTC(poptorch::bindings::importPoptorchMetadataFromFile));
  m.def("execute", PTC(poptorch::bindings::execute));
  m.def("updateOptimizers", PTC(poptorch::bindings::updateOptimizers));
  m.def("getTimestamps", PTC(poptorch::bindings::getTimestamps));
  m.def("readOptimizerState", PTC(poptorch::bindings::readOptimizerState));
  m.def("setRngState", PTC(poptorch::bindings::setRngState));
  m.def("getRngState", PTC(poptorch::bindings::getRngState));
  m.def("getRandomSeed", PTC(poptorch::bindings::getRandomSeed));
  m.def("writeOptimizerState", PTC(poptorch::bindings::writeOptimizerState));
  m.def("loadEngineAndConnectStreams",
        PTC(poptorch::bindings::loadEngineAndConnectStreams));
  m.def("copyWeightsToDevice_impl",
        PTC(poptorch::bindings::copyWeightsToDeviceImpl));
  m.def("copyNamedBuffersToDevice_impl",
        PTC(poptorch::bindings::copyNamedBuffersToDeviceImpl));
  m.def("copyWeightsToHost_impl",
        PTC(poptorch::bindings::copyWeightsToHostImpl));
  m.def("ipuHardwareVersion",
        PTC(poptorch::popart_compiler::ipuHardwareVersion),
        py::arg("numIpus") = 1);
  m.def("setCustomCodeletsPath",
        PTC(poptorch::popart_compiler::setCustomCodeletsPath));
  m.def("setLogLevel", PTC(poptorch::bindings::setLogLevel),
        py::arg("level") = 2);
  m.def("setPopartLogLevel", PTC(poptorch::bindings::setPopartLogLevelUInt));
  m.def("_getPopartIR", PTC(poptorch::bindings::getPopartIR));
  m.def("_getTensorNames", PTC(poptorch::bindings::getTensorNames));
  m.def("detachFromDevice", PTC(poptorch::bindings::detachFromDevice));
  m.def("attachToDevice", PTC(poptorch::bindings::attachToDevice));
  m.def("isAttachedToDevice", PTC(poptorch::bindings::isAttachedToDevice));
  m.def("registerCPUCallBack", PTC(poptorch::registerCPUCallBack));
  m.def("isAlreadyRegistered", PTC(poptorch::alreadyRegistered));
  m.def("registerBuffersWithCallback",
        PTC(poptorch::registerBuffersWithCallback));
  m.def("_validateOptions", PTC(poptorch::parseSessionOptionsVoid));

  py::enum_<poptorch::TracingMode>(m, "TracingMode")
      .value("PopART", poptorch::TracingMode::POPART)
      .export_values();

  m.def("poptorchAtExit", PTC(poptorch::poptorchAtExit));
  m.def("destroyDispatcher", PTC(poptorch::destroyDispatcher));
  m.def("startDispatch", PTC(poptorch::startDispatch));
  m.def("isCompilingWithDispatcher", PTC(poptorch::isCompilingWithDispatcher));
  m.def("endDispatch", PTC(poptorch::endDispatch));
  m.def("startParametersMove", PTC(poptorch::startParametersMove));
  m.def("endParametersMove", PTC(poptorch::endParametersMove));
  m.def("startOutputsMove", PTC(poptorch::startOutputsMove));
  m.def("endOutputsMove", PTC(poptorch::endOutputsMove));
  m.def("createGraph", PTC(poptorch::createGraph));
  m.def("mapParamsToNames", PTC(poptorch::mapParamsToNames));
  m.def("setPerReplica", PTC(poptorch::setPerReplica));
  m.def("finalizeGraph", PTC(poptorch::finalizeGraph));
  m.def("compileWithManualTracing",
        PTC(poptorch::bindings::compileWithManualTracing));
  m.def("processDispatchAndImportExecutable",
        PTC(poptorch::bindings::processDispatchAndImportExecutable));
  m.def("_throwTestError", PTC(poptorch::popart_compiler::throwTestError));
  m.def("getIpuTensorId", PTC(poptorch::getIpuTensorId));

  poptorch::bindings::initialiseExceptionHandling(m);

  py::enum_<poptorch::popart_compiler::TestErrorType>(m, "TestErrorType")
      .value("Poptorch", poptorch::popart_compiler::TestErrorType::Poptorch)
      .value("Popart", poptorch::popart_compiler::TestErrorType::Popart)
      .value("PopartInternal",
             poptorch::popart_compiler::TestErrorType::PopartInternal)
      .value("Poplibs", poptorch::popart_compiler::TestErrorType::Poplibs)
      .value("PoplarUnrecoverable",
             poptorch::popart_compiler::TestErrorType::PoplarUnrecoverable)
      .value("PoplarUnknown",
             poptorch::popart_compiler::TestErrorType::PoplarUnknown)
      .value(
          "PoplarRecoverableFullReset",
          poptorch::popart_compiler::TestErrorType::PoplarRecoverableFullReset)
      .value("PoplarLinkError",
             poptorch::popart_compiler::TestErrorType::PoplarLinkError);

  py::register_exception_translator(
      [](std::exception_ptr p) { // NOLINT: Don't change 'p' to a const&
        try {
          if (p) {
            std::rethrow_exception(p);
          }
        } catch (const poptorch::PoptorchError &e) {
          e.setErrorIndicator();
        }
      });
}
