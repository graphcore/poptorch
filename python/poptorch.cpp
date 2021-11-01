// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/script.h>

#include <limits>
#include <sstream>
#include <unordered_map>

#include "popart_compiler/Compiler.hpp"

// Shared enums across the ABI boundary.
#include "popart_compiler/PopartEnums.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/AliasProcessing.hpp"
#include "poptorch/AutomaticCasting.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/ImplicitCasting.hpp"
#include "poptorch/InplaceOps.hpp"
#include "poptorch/LowerToPopart.hpp"
#include "poptorch/OverlappedIO.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

// This is needed to catch STL exceptions and attach some context to them.
#define CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION                                \
  catch (const poptorch::logging::Error &e) {                                  \
    throw poptorch::ExceptionInfo(e, "poptorch_cpp_error", e.file(),           \
                                  e.line());                                   \
  }                                                                            \
  catch (const std::out_of_range &e) {                                         \
    throw poptorch::ExceptionInfo(e, "std::out_of_range", __FILE__, __LINE__); \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    throw poptorch::ExceptionInfo(e, nullptr, __FILE__, __LINE__);             \
  }

void beginIpuBlock(int64_t stage_id, int64_t phase_id, int64_t ipu_id) {
  UNUSED(stage_id);
  UNUSED(phase_id);
  UNUSED(ipu_id);
}

void whileLoopBegin(const at::Tensor &condition,           // NOLINT
                    const c10::List<at::Tensor> &inputs) { // NOLINT
  UNUSED(condition);
  UNUSED(inputs);
}

at::Tensor castOp(at::Tensor tensor, std::string &&type) {
  UNUSED(type);

  // If the type to cast to is f16 then we need to cast to f32. The reason being
  // is that by default we will just ignore the type, however this will only
  // work if the original type was f32.

  // Consider:
  /* MyTensor = MyTensor.as(INT8)

     MyTensor = MyTensor.half() # Convert to half.

     out = conv(MyTensor) # This would be an illegal INT8 convolution.
  */
  if (type == "FLOAT16" || type == "FLOAT32") {
    return tensor.to(at::ScalarType::Float);
  }
  return tensor;
}

at::Tensor setAvailableMemory(at::Tensor t, double mem) {
  UNUSED(mem);
  return t;
}

at::Tensor ipuPrintTensor(const at::Tensor &t, std::string &&title) {
  UNUSED(title);
  return t.clone();
}

at::Tensor setMatMulSerialization(at::Tensor matmul, std::string mode, // NOLINT
                                  int64_t factor, bool keep_precision) {
  UNUSED(mode);
  UNUSED(factor);
  UNUSED(keep_precision);
  return matmul;
}

at::Tensor setOverlapForInput(at::Tensor t, const std::string &mode) {
  UNUSED(mode);
  return t;
}

at::Tensor setOverlapForOutput(at::Tensor t, const std::string &mode) {
  UNUSED(mode);
  return t;
}

at::Tensor identityOp(const at::Tensor &t) { return t.clone(); }

at::Tensor identityLoss(const at::Tensor &t, int64_t reduction) {
  constexpr int64_t sum = 0;
  constexpr int64_t mean = 1;
  constexpr int64_t none = 2;

  if (reduction == sum) {
    return at::sum(t);
  }

  if (reduction == mean) {
    return at::mean(t);
  }

  ERROR_ON(reduction != none);
  return t.clone();
}

c10::List<at::Tensor>
customOperation(c10::List<at::Tensor> inputs,          // NOLINT
                std::string name, std::string domain,  // NOLINT
                int64_t version, int64_t num_outputs,  // NOLINT
                c10::List<at::Tensor> example_outputs, // NOLINT
                std::string attributes_map_id) {       // NOLINT
  UNUSED(inputs);
  UNUSED(name);
  UNUSED(domain);
  UNUSED(version);
  UNUSED(num_outputs);
  UNUSED(attributes_map_id);

  return example_outputs;
}

c10::List<at::Tensor> ctcBeamSearchDecoder(const at::Tensor &log_probs,
                                           const at::Tensor &lengths,
                                           int64_t blank, int64_t width,
                                           int64_t top_paths) {
  UNUSED(lengths);
  UNUSED(blank);
  UNUSED(width);

  ERROR_ON_MSG(log_probs.sizes().size() != 3,
               "Incorrect shape for first input to CTC beam search decoder.");
  unsigned input_len = log_probs.sizes()[0];
  unsigned batch_size = log_probs.sizes()[1];

  at::Tensor path_probs = at::zeros({batch_size, top_paths});
  at::Tensor path_lens = at::zeros({batch_size, top_paths});
  at::Tensor decoded_paths = at::zeros({batch_size, top_paths, input_len});

  return c10::List<at::Tensor>({path_probs, path_lens, decoded_paths});
}

void callCpuOp(const c10::List<at::Tensor> &inputs, const std::string &name) {
  UNUSED(inputs);
  UNUSED(name);
}

c10::List<at::Tensor> endCPUOp(c10::List<at::Tensor> output) {
  UNUSED(output);
  return output;
}

void setAttribute(const std::string &attribute, const std::string &key,
                  const std::string &value) {
  UNUSED(attribute);
  UNUSED(key);
  UNUSED(value);
}

void clearAttribute(const std::string &attribute, const std::string &key) {
  UNUSED(attribute);
  UNUSED(key);
}

// We track the outputs of the if in this brach as it is easier to add them
// immediately before.
void elseBranch(c10::List<at::Tensor> if_out) { // NOLINT
  UNUSED(if_out);
}

c10::List<at::Tensor> endIf(at::Tensor condition,               // NOLINT
                            c10::List<at::Tensor> example_outs, // NOLINT
                            c10::List<at::Tensor> else_out) {   // NOLINT
  UNUSED(condition);
  UNUSED(else_out);
  return example_outs;
}

void startForLoop(c10::List<at::Tensor> inputs) { // NOLINT
  UNUSED(inputs);
}

c10::List<at::Tensor>
endForLoop(c10::List<at::Tensor> outputs,               // NOLINT
           c10::List<at::Tensor> inputs, int64_t count, // NOLINT
           c10::List<at::Tensor> example_outputs) {     // NOLINT
  UNUSED(count);
  UNUSED(outputs);
  UNUSED(inputs);
  return example_outputs;
}

void optimizerGroup(int64_t group, c10::List<at::Tensor> &&inputs) {
  UNUSED(group);
  UNUSED(inputs);
}

void endMultiConv(
    c10::optional<c10::List<double>> &&available_memory_proportions,
    c10::optional<c10::List<int64_t>> &&partials_types,
    c10::optional<int64_t> &&plan_type,
    c10::optional<int64_t> &&per_conv_reserved_tiles,
    c10::optional<double> &&cycle_back_off,
    c10::optional<c10::List<int64_t>> &&enableConvDithering) {
  UNUSED(available_memory_proportions);
  UNUSED(partials_types);
  UNUSED(plan_type);
  UNUSED(per_conv_reserved_tiles);
  UNUSED(cycle_back_off);
  UNUSED(enableConvDithering);
}

void pushNameScope(const std::string &&name) { UNUSED(name); }

// Many operations just appear as having no return or inputs but our passes use
// them to derive user intent. They can just use this implementation.
void nullOp() {}

static auto registry =
    torch::RegisterOperators("poptorch::begin_ipu_block", &beginIpuBlock)
        .op("poptorch::end_ipu_block", &nullOp)
        .op("poptorch::ipu_print_tensor", &ipuPrintTensor)
        .op("poptorch::internal_cast", &castOp)
        .op("popart::nop", &identityOp)
        .op("poptorch::custom_operation", &customOperation)
        .op("poptorch::ctc_beam_search_decoder", &ctcBeamSearchDecoder)
        .op("poptorch::identity_loss", &identityLoss)
        .op("poptorch::while_loop_begin", &whileLoopBegin)
        .op("poptorch::end_loop_begin", &nullOp)
        .op("poptorch::start_if_true", &nullOp)
        .op("poptorch::start_if_false", &elseBranch)
        .op("poptorch::end_if", &endIf)
        .op("poptorch::start_for_loop", &startForLoop)
        .op("poptorch::end_for_loop", &endForLoop)
        .op("poptorch::optimizer_group", &optimizerGroup)
        .op("poptorch::set_matmul_serialization", &setMatMulSerialization)
        .op("poptorch::set_overlap_for_input", &setOverlapForInput)
        .op("poptorch::set_overlap_for_output", &setOverlapForOutput)
        .op("poptorch::recomputation_checkpoint", &identityOp)
        .op("poptorch::set_available_memory", &setAvailableMemory)
        .op("poptorch::begin_multi_conv", &nullOp)
        .op("poptorch::end_multi_conv", &endMultiConv)
        .op("poptorch::push_name_scope", &pushNameScope)
        .op("poptorch::pop_name_scope", &nullOp)
        .op("poptorch::begin_autocast", &nullOp)
        .op("poptorch::suppress_autocast", &nullOp)
        .op("poptorch::restore_autocast", &nullOp)
        .op("poptorch::end_cpu_op", &endCPUOp)
        .op("poptorch::call_cpu_op", &callCpuOp)
        .op("poptorch::set_attribute", setAttribute)
        .op("poptorch::clear_attribute", &clearAttribute);

namespace poptorch {
namespace {

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
  try {
    auto itr = callbacks.find(ID);

    ERROR_ON_MSG(itr == callbacks.end(), "Callback has not been registered.");

    CallbackMetadata &metadata = itr->second;

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
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

// Python interface to map a given CPU op with the IR calls.
void registerCPUCallBack(const py::object &obj, const std::string &ID) {
  // Skip if we've already added a callback for this function.
  if (callbacks.find(ID) != callbacks.end()) {
    return;
  }

  // Structure to store the information given by python to be forwarded to the
  // backend.
  CallbackMetadata metadata;

  // Wrap that in a lambda so we don't have to expose the naked pytorch function
  // pointer thing.
  metadata.the_callback = [=]() {
    // We wrap the user call in a function called "execute"
    obj.attr("execute")();
  };

  metadata.buffer_registration_callback = [=]() {
    obj.attr("registerPersistentData")();
  };

  // Map the string identifier to the metadata.
  callbacks.insert({ID, metadata});
}

void initCallbackBuffers() {
  for (auto &pair : callbacks) {
    pair.second.buffer_registration_callback();
  }
}

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

void copyParametersDict(Optimizer *out, const py::dict &in) {
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
std::vector<Optimizer> parseOptimizer(const py::dict &opt) {
  if (opt.empty()) {
    return {};
  }

  OptimizerType type = OptimizerType::NONE;
  std::uint64_t num_groups;
  type = static_cast<OptimizerType>(
      getOptimizerValue<std::uint64_t>(opt, "optimizer_type"));
  auto defaults = getOptimizerValue<py::dict>(opt, "defaults");
  auto groups = getOptimizerValue<py::list>(opt, "groups");
  num_groups = groups.size();
  std::vector<Optimizer> optimizers;
  // Note: all the group variables and optimizer variables are
  // automatically forwarded to the Compiler backend however
  // the optimizer attributes are extracted here.
  bool use_tf_variant = false;
  if (type == OptimizerType::RMSPROP ||
      type == OptimizerType::RMSPROP_CENTERED) {
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

    if (type == OptimizerType::SGD1 || type == OptimizerType::SGD2) {
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

  copyParametersDict(&optimizers[0], defaults);
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
  torch::jit::Stack stack = torch::jit::toTraceableStack(tensors);
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

std::string castToString(py::handle obj) {
  if (py::isinstance<py::str>(obj)) {
    return obj.cast<std::string>();
  }
  if (py::isinstance<py::int_>(obj)) {
    std::stringstream ss;
    ss << obj.cast<std::uint64_t>();
    return ss.str();
  }
  ERROR("Don't know how to convert type " << obj.get_type() << " to string");
}

void parseAnchors(AnchorList *map, const py::list &list) {
  for (auto elem : list) {
    auto anchor = elem.cast<py::list>();
    Anchor a(anchor[0].cast<std::string>(), anchor[2].cast<std::uint64_t>(),
             anchor[3].cast<std::uint64_t>());
    map->push_back(a);
  }
}

SessionOptions parseSessionOptions(const py::dict &opt) {
  logging::LogContext ctx_func("parseSessionOptions");
  // steps, replicationFactor, profile
  SessionOptions options;

  for (auto element : opt) {
    auto option_name = element.first.cast<std::string>();
    logging::LogContext ctx("option: " + option_name);
    // Exception patterns_level is handled at the same time as "patterns"
    if (option_name == "patterns_level" || option_name == "anchored_tensors") {
      continue;
    }
    if (option_name == "compilation_progress_bar_fn") {
      options.setCompilationProgressLogger(element.second.cast<py::function>());
    } else if (py::isinstance<py::bool_>(element.second)) {
      options.addBoolOption(option_name.c_str(), element.second.cast<bool>());
    } else if (py::isinstance<py::float_>(element.second)) {
      options.addDoubleOption(option_name.c_str(),
                              element.second.cast<double>());
    } else if (py::isinstance<py::int_>(element.second)) {
      options.addUint64Option(option_name.c_str(),
                              element.second.cast<std::uint64_t>());
    } else if (py::isinstance<py::str>(element.second)) {
      options.addStringOption(option_name.c_str(),
                              element.second.cast<std::string>().c_str());
    } else if (py::isinstance<py::set>(element.second) ||
               py::isinstance<py::list>(element.second)) {
      for (auto option : element.second.cast<py::list>()) {
        options.insertStringOption(option_name.c_str(),
                                   castToString(option).c_str());
      }
    } else if (py::isinstance<py::dict>(element.second)) {
      const std::string &id = option_name;

      if (id == "available_memory_proportion") {
        for (auto option : element.second.cast<py::dict>()) {
          options.setMemoryProportion(option.first.cast<std::uint64_t>(),
                                      option.second.cast<float>());
        }
      } else if (id == "patterns") {
        options.setPatternsLevel(
            opt.cast<py::dict>()["patterns_level"].cast<std::uint64_t>());

        for (auto option : element.second.cast<py::dict>()) {
          options.addPattern(option.first.cast<std::string>().c_str(),
                             option.second.cast<bool>());
        }
      } else if (id.rfind("location_", 0) == 0) {
        for (auto option : element.second.cast<py::dict>()) {
          options.setTensorLocation(id.c_str(),
                                    option.first.cast<std::string>().c_str(),
                                    option.second.cast<std::uint64_t>());
        }
      } else {
        for (auto option : element.second.cast<py::dict>()) {
          options.insertStringPairOption(
              id.c_str(), option.first.cast<std::string>().c_str(),
              castToString(option.second).c_str());
        }
      }
    } else {
      ERROR("Unknown option type " << element.second.get_type());
    }
  }
  return options;
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

// Recursively print the input type, which may be:
// - A tensor
// - A tuple (Which may contain tensors or tuples)
template <typename TensorIt>
void printOrigType(std::ostream &os, const c10::TypePtr &type,
                   TensorIt &current) {
  auto tuple_type = type->cast<c10::TupleType>();
  if (tuple_type != nullptr) {
    // Print and recurse through tuples
    os << '(';
    for (std::size_t i = 0; i < tuple_type->containedTypes().size(); i++) {
      auto t = tuple_type->containedTypes()[i];
      printOrigType(os, t, current);
      if (i != tuple_type->containedTypes().size() - 1) {
        os << ", ";
      }
    }
    os << ')';
    return;
  }
  auto tensor_type = type->cast<c10::TensorType>();
  if (tensor_type != nullptr) {
    // Recursion base case, just print the original tensor type
    os << *tensor_type->withScalarType(current->scalar_type());
    // Advance the current tensor iterator so that the next invocation
    // uses the correct tensor
    std::advance(current, 1);
    return;
  }
  ERROR("Invalid type being traced, expected tensors or tuples");
}

// Print the graph input string which matches that of Graph::print
void printGraphInputStr(std::ostream &os, const torch::jit::Graph &graph,
                        const std::vector<at::Tensor> &input_tensors,
                        bool is_trace_input) {
  bool first = true;
  auto it_tensors = input_tensors.begin();
  for (const auto *input : graph.inputs()) {
    if (!first) {
      os << ",\n      ";
    }
    first = false;
    os << "%" << input->debugName();
    os << " : ";
    // After reaching end(input_tensors), the remaining
    // (N - size(input_tensors)) graph inputs will have the same traced
    // type so we don't need to process those
    if (is_trace_input || it_tensors == input_tensors.end()) {
      os << *input->type();
    } else {
      printOrigType(os, input->type(), it_tensors);
    }
  }
}

// Prints the graph out to the log (trace), print both the trace inputs and
// actual inputs if trace_input_str is not empty
void logGraph(const char *intro_str, const torch::jit::Graph &graph,
              bool has_converted_any_half,
              const std::vector<at::Tensor> &input_tensors) {
  std::ostringstream graph_str;
  graph_str << intro_str << "\n";

  // If there are no halves converted to floats, simply print the graph
  if (!has_converted_any_half) {
    graph_str << graph;
    logging::trace("{}", graph_str.str());
    return;
  }

  // Print the trace inputs
  graph_str << "graph(";
  printGraphInputStr(graph_str, graph, input_tensors, true);
  graph_str << "):\n";

  // Print the original inputs
  graph_str << "[orig:";
  printGraphInputStr(graph_str, graph, input_tensors, false);
  graph_str << "]\n";

  std::vector<const torch::jit::Node *> groups;
  for (const auto *n : graph.nodes()) {
    n->print(graph_str, 1, &groups, true);
  }
  graph_str << "  return (" << graph.outputs() << ")\n";
  size_t i = 0;

  for (const auto *fg : groups) {
    graph_str << "with " << fg->kind().toQualString() << "_" << i++ << " = "
              << *fg->g(torch::jit::attr::Subgraph);
  }
  logging::trace("{}", graph_str.str());
}

// Prints the graph but substitutes BFloat16 for Float16/Float32
void printGraphBeforeHalfFloatResolution(const torch::jit::Graph &graph) {
  std::ostringstream graph_oss;
  graph_oss << graph;
  std::string graph_str = graph_oss.str();

  size_t start = 0;
  const std::string from = "BFloat16";
  const std::string to = "Float16/Float32";
  while ((start = graph_str.find(from, start)) != std::string::npos) {
    graph_str.replace(start, from.length(), to);
    start += to.length();
  }

  logging::trace("Graph right before half/float resolution:\n{}", graph_str);
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

poptorch::LowerToPopart lowerToPopartFromTrace(
    py::handle h, const pybind11::dict &python_traced_params,
    const pybind11::tuple &inputs, bool has_converted_any_half,
    const pybind11::dict &options, bool training,
    const py::dict &optimizer_dict, const py::function &attribute_accessor,
    const bool added_dummy_output, const py::list &anchors,
    const py::dict &model_parameters) {
  auto *module = asModule(h);

  auto forward = module->get_method("forward");
  auto graph_and_tensors =
      torch::jit::LowerGraph(*forward.graph(), module->_ivalue());
  auto graph = graph_and_tensors.first;

  // If we added a dummy output to make tracing work (no output case), remove
  // it
  if (added_dummy_output) {
    ERROR_ON(graph->outputs().size() != 1);
    graph->eraseOutput(0);
  }

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack input_stack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> input_tensors;
  for (const torch::jit::IValue &value : input_stack) {
    buildTensorList(value, &input_tensors);
  }

  // This is subset of "parameters_tensors", but only those actually used.
  // The order matches the that of graph inputs, and the last
  // traced_parameter_tensors.size() graph inputs matches these, while those
  // which come before are the "actual" inputs.
  std::vector<at::Tensor> traced_parameter_tensors;
  for (const torch::jit::IValue &value : graph_and_tensors.second) {
    buildTensorList(value, &traced_parameter_tensors, true);
  }

  identifyZeroSizedTensors(input_tensors);
  identifyZeroSizedTensors(traced_parameter_tensors);

  std::vector<std::string> parameters =
      getParameterNames(python_traced_params, traced_parameter_tensors);

  traced_parameter_tensors = remapTensors(
      python_traced_params, model_parameters, traced_parameter_tensors);

  std::vector<Optimizer> optimizers = parseOptimizer(optimizer_dict);
  SessionOptions parsed_options = parseSessionOptions(options);

  logGraph("Lowered graph:", *graph, has_converted_any_half, input_tensors);

  torch::jit::EliminateDeadCode(graph);
  torch::jit::PeepholeOptimize(graph);
  torch::jit::EliminateDeadCode(graph);

  torch::jit::LowerSimpleTuples(graph);
  torch::jit::PeepholeOptimize(graph);

  logGraph("Graph before attributising IO overlap specifiers", *graph,
           has_converted_any_half, input_tensors);
  poptorch::attributiseOverlappedIO(graph.get());

  logGraph("Graph before handling aliases:", *graph, has_converted_any_half,
           input_tensors);

  poptorch::resolveAliases(graph.get());

  logGraph("Graph before handling inplace ops:", *graph, has_converted_any_half,
           input_tensors);

  // Ensure that list elements have the ListTypeWithNumElements type which
  // contains the number of elements in a list
  poptorch::type_and_constant_canonicalization::addListNumElements(graph.get());

  auto inplace_op_handler = std::make_shared<InplaceOpHandler>(
      graph, traced_parameter_tensors.size(), anchors.size(),
      parsed_options.replicationFactor() > 1 &&
          parsed_options.broadcastBuffers());

  // Any types with ListTypeWithNumElements must be revereted (revert = true)
  // to allow constant evaluation to proceed
  poptorch::type_and_constant_canonicalization::addListNumElements(graph.get(),
                                                                   true);

  logGraph("Graph right before evaluating constant expressions:", *graph,
           has_converted_any_half, input_tensors);

  poptorch::type_and_constant_canonicalization::evaluateConstexprs(graph.get());

  logGraph("Graph right before casting making integer params as constant "
           "inputs:",
           *graph, has_converted_any_half, input_tensors);

  poptorch::type_and_constant_canonicalization::makeConstantIntParams(
      graph.get(), parameters, traced_parameter_tensors);

  logGraph("Graph right before casting unsupported inputs:", *graph,
           has_converted_any_half, input_tensors);
  poptorch::type_and_constant_canonicalization::castUnsupportedInputs(
      graph.get());

  logGraph("Graph right before output type changes:", *graph,
           has_converted_any_half, input_tensors);
  poptorch::type_and_constant_canonicalization::checkAndChangeOutputTypes(
      graph.get());

  logGraph("Graph right before constant canonicalisation:", *graph,
           has_converted_any_half, input_tensors);
  poptorch::type_and_constant_canonicalization::canonicaliseConstants(
      graph.get());

  // After constant canonicalisation, it is safe to ensure that list have
  // ListTypeWithNumElements once again, plus the last step will have added
  // new list constructs.
  poptorch::type_and_constant_canonicalization::addListNumElements(graph.get());

  // Convert the IR to half to match the inputs/actual usage.
  logGraph("Graph before canonicalising half:", *graph, has_converted_any_half,
           input_tensors);
  poptorch::canonicaliseHalfInputs(graph.get(), input_tensors,
                                   traced_parameter_tensors);

  logging::trace("Graph right before canonicalizing lists:\n{}", *graph);

  poptorch::canonicalizeLists(graph.get());

  logging::trace("Graph right before automatic casting:\n{}", *graph);

  poptorch::automaticCasting(graph.get());

  poptorch::cpuOffloadingCleanup(graph.get());

  logging::trace("Graph right before canonicalization:\n{}", *graph);
  // Convert any unsupported ATEN nodes in the graph to a popart
  // representation.
  poptorch::canonicalize(graph.get());

  printGraphBeforeHalfFloatResolution(*graph);

  poptorch::annotateSubgraphs(graph.get());

  poptorch::resolveHalfOrFloat(graph.get());

  // Enforce any constraints that aren't enforced by popart.
  poptorch::canonicalizeLate(graph.get());

  logging::trace("Graph right after canonicalization:\n{}", *graph);

  if (training) {
    poptorch::removeSurplusIdentityLosses(graph.get());
    poptorch::addDetachOperations(graph.get());
    logging::trace("Graph right after add detach operations:\n{}", *graph);
  }

  // Warn the user if any operations couldn't be canonicalised.
  poptorch::warnOnUnsupportedAten(graph.get());

  logging::trace("Graph right before popart:\n{}", *graph);

  // Get the callback buffers from python, we have to do this at the last
  // possible moment due to tracing.
  initCallbackBuffers();

  AnchorList anchors_list;
  parseAnchors(&anchors_list, anchors);

  poptorch::LowerToPopart lower(
      graph.get(), std::move(traced_parameter_tensors), std::move(parameters),
      inplace_op_handler, training, std::move(optimizers), parsed_options,
      attribute_accessor, callbacks, std::move(anchors_list));
  lower.lower(&input_tensors);

  // Clear the callbacks after compilation.
  callbacks.clear();

  return lower;
}

} // namespace

void copyWeightsToHostImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors) {
  // Copy the weights or warn if this is before first time compilation.
  try {
    if (!executable) {
      logging::warn(
          "Call to copyWeightsToHost ignored as model has not been compiled "
          "(PopTorch will compile models on first invocation).");
    } else {
      executable->copyWeightsToHost(
          getParameterBuffers(parameter_names, parameter_tensors));
    }
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void copyWeightsToDeviceImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors) {
  // Copy the weights or warn if this is before first time compilation.
  try {
    if (!executable) {
      logging::warn(
          "Call to copyWeightsToDevice ignored as model has not been compiled "
          "(PopTorch will compile models on first invocation).");
    } else {
      executable->copyWeightsToDevice(
          getParameterBuffers(parameter_names, parameter_tensors));
    }
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

std::string
getPopartIR(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  try {
    return executable->getPopartIR();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void detachFromDevice(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  try {
    executable->detachFromDevice();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void attachToDevice(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  try {
    executable->attachToDevice();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

bool isAttachedToDevice(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  try {
    return executable->isAttachedToDevice();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void setLogLevel(std::uint64_t level) {
  ERROR_ON(level > static_cast<std::uint64_t>(logging::Level::Off) ||
           level == 5);
  logging::setLogLevel(static_cast<logging::Level>(level));
}

void setPopartLogLevelUInt(std::uint64_t level) {
  ERROR_ON(level > static_cast<std::uint64_t>(logging::Level::Off) ||
           level == 5);
  poptorch::setPopartLogLevel(static_cast<logging::Level>(level));
}

void loadEngineAndConnectStreams(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  try {
    executable->loadEngineAndConnectStreams();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

std::vector<pybind11::object>
execute(const std::shared_ptr<poptorch::PoplarExecutable> &executable,
        const pybind11::tuple &inputs, py::dict *optimizer_dict) {
  try {
    // Create a jit stack from the incoming pytorch tensors.
    torch::jit::Stack input_stack = torch::jit::toTraceableStack(inputs);

    // And turn convert them into at tensors which we can then resolve the
    // address of.
    std::vector<at::Tensor> input_tensors;
    for (const torch::jit::IValue &value : input_stack) {
      buildTensorList(value, &input_tensors);
    }

    // Create an empty optimizer for inference, this will not be applied.
    std::vector<Optimizer> optimizers;

    if (optimizer_dict != nullptr) {
      optimizers = parseOptimizer(*optimizer_dict);
    }

    std::vector<at::IValue> output_tensors =
        executable->run(&input_tensors, optimizers);

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
      case OutputElemType::Tensor: {
        ERROR_ON_MSG(tensor_it == output_tensors.end(),
                     "Not enough tensors to unpack");
        auto object = torch::jit::toPyObject(*tensor_it);
        tensor_it++;
        return object;
      }
      case OutputElemType::Tuple: {
        std::int64_t num_elements = type_it->num_elements;
        pybind11::tuple pytuple(num_elements);
        for (std::int64_t i = 0; i < num_elements; ++i) {
          type_it++;
          pytuple[i] = process_output();
        }
        return std::move(pytuple);
      }
      case OutputElemType::List: {
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
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

py::dict readOptimizerState(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  py::dict optim_state;
  py::dict state_tensors;
  py::dict param_tensors;
  try {
    const auto &compiler = executable->getCompiler();
    std::vector<TensorMetadata> metadata_list =
        compiler.optimizerTensorMetadataList();

    std::vector<void *> host_buffers;
    for (const TensorMetadata &meta : metadata_list) {
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
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
  optim_state["state"] = std::move(state_tensors);
  optim_state["param"] = std::move(param_tensors);
  return optim_state;
}

std::vector<pybind11::object>
getTimestamps(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  const auto &compiler = executable->getCompiler();
  auto num_inputs = compiler.getNumInputs();
  auto num_outputs = compiler.getNumOutputs();

  py::list input;
  py::list input_complete;
  py::list output;
  py::list output_complete;

  try {
    for (size_t i = 0; i < num_inputs; ++i) {
      input.append(py::cast(compiler.getInputTimestamps(i)));
      input_complete.append(py::cast(compiler.getInputCompleteTimestamps(i)));
    }

    for (size_t i = 0; i < num_outputs; ++i) {
      output.append(py::cast(compiler.getOutputTimestamps(i)));
      output_complete.append(py::cast(compiler.getOutputCompleteTimestamps(i)));
    }
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION

  return {input, input_complete, output, output_complete};
}

void processPrecisionOptions(py::handle h) {
  try {
    auto values_dict = h.attr("_values").cast<py::dict>();

    poptorch::setAutocastEnabled(values_dict["autocast_enabled"].cast<bool>());

    auto policy = values_dict["autocast_policy_dict"].cast<py::dict>();
    setAutocastHalf(policy["fp16"].cast<std::vector<std::string>>());
    setAutocastFloat(policy["fp32"].cast<std::vector<std::string>>());
    setAutocastPromote(policy["promote"].cast<std::vector<std::string>>());
    setAutocastDemote(policy["demote"].cast<std::vector<std::string>>());

    poptorch::setHalfFloatCastingBehavior(static_cast<HalfFloatCasting>(
        values_dict["half_float_casting"].cast<uint64_t>()));

    poptorch::setRunningStatisticsAlwaysFloat(
        values_dict["running_statistics_always_float"].cast<bool>());
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

bool pyIsGraphNondeterministic(py::handle h) {
  try {
    auto *module = asModule(h);
    auto forward = module->get_method("forward");
    auto graph_and_tensors =
        torch::jit::LowerGraph(*forward.graph(), module->_ivalue());
    auto graph = graph_and_tensors.first;
    const auto &nodes = graph->nodes();
    return std::any_of(nodes.begin(), nodes.end(),
                       [](const torch::jit::Node *n) {
                         return poptorch::isNondeterministic(*n);
                       });
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void compileWithTraceAndExport(
    py::handle h, const pybind11::dict &python_traced_params,
    const pybind11::tuple &inputs, bool has_converted_any_half,
    const pybind11::dict &options, bool training,
    const py::dict &optimizer_dict, const py::function &attribute_accessor,
    const bool added_dummy_output, const py::list &anchors,
    const py::dict &model_parameters, const std::string &export_filename) {
  try {
    auto lower = lowerToPopartFromTrace(
        h, python_traced_params, inputs, has_converted_any_half, options,
        training, optimizer_dict, attribute_accessor, added_dummy_output,
        anchors, model_parameters);
    return lower.compileAndExport(export_filename);
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

uint64_t
cycleCount(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  try {
    return executable->getCompiler().getCycleCount();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

std::shared_ptr<poptorch::PoplarExecutable> processTraceAndImportExecutable(
    py::handle h, const pybind11::dict &python_traced_params,
    const pybind11::tuple &inputs, bool has_converted_any_half,
    const pybind11::dict &options, bool training,
    const py::dict &optimizer_dict, const py::function &attribute_accessor,
    const bool added_dummy_output, const py::list &anchors,
    const py::dict &model_parameters, const std::string &import_filename,
    std::int64_t offset) {
  try {
    auto lower = lowerToPopartFromTrace(
        h, python_traced_params, inputs, has_converted_any_half, options,
        training, optimizer_dict, attribute_accessor, added_dummy_output,
        anchors, model_parameters);
    return lower.loadExecutableFromFile(import_filename, offset);
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithTrace(py::handle h, const pybind11::dict &python_traced_params,
                 const pybind11::tuple &inputs, bool has_converted_any_half,
                 const pybind11::dict &options, bool training,
                 const py::dict &optimizer_dict,
                 const py::function &attribute_accessor,
                 const bool added_dummy_output, const py::list &anchors,
                 const py::dict &model_parameters) {
  try {
    auto lower = lowerToPopartFromTrace(
        h, python_traced_params, inputs, has_converted_any_half, options,
        training, optimizer_dict, attribute_accessor, added_dummy_output,
        anchors, model_parameters);
    return lower.compile();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void throwTestErrorSafe(TestErrorType type) {
  try {
    poptorch::throwTestError(type);
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

std::shared_ptr<poptorch::PoplarExecutable> compileWithManualTracing(
    std::vector<at::Tensor> &inputs, const std::vector<at::Tensor> &parameters,
    const std::vector<std::string> &parameter_names,
    const pybind11::dict &options, const py::function &attribute_accessor) {
  try {
    logging::debug("Compile with manual tracing");
    std::shared_ptr<torch::jit::Graph> graph = getTracedGraph();

    logging::debug("Traced graph: {}", *graph);

    AnchorList anchors_list;
    std::vector<Optimizer> optimizers;

    auto inplace_op_handler =
        std::make_shared<InplaceOpHandler>(graph, 0, 0, true);

    poptorch::LowerToPopart lower(
        graph.get(), parameters, parameter_names, inplace_op_handler, false,
        std::move(optimizers), parseSessionOptions(options), attribute_accessor,
        callbacks, std::move(anchors_list));

    lower.lower(&inputs);

    return lower.compile();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

} // namespace poptorch

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
  void setWhat(const std::string &message) {
    PyErr_SetString(m_ptr, message.c_str());
  }

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
};

class RecoverableError : public Error {
public:
  using Error::Error;

  void setRecoveryAction(const std::string &recoveryAction) {
    py::object x = py::cast(recoveryAction);
    PyObject_SetAttrString(m_ptr, "recovery_action", x.ptr());
  }
};

PYBIND11_MODULE(poptorch_core, m) { // NOLINT
  py::class_<poptorch::PoplarExecutable,
             std::shared_ptr<poptorch::PoplarExecutable>>
      give_me_a_name(m, "InternalPoplarExecutable");

  m.def("processPrecisionOptions", poptorch::processPrecisionOptions);
  m.def("isGraphNondeterministic", poptorch::pyIsGraphNondeterministic);
  m.def("compileWithTrace", poptorch::compileWithTrace);
  m.def("compileWithTraceAndExport", poptorch::compileWithTraceAndExport);
  m.def("cycleCount", poptorch::cycleCount);
  m.def("processTraceAndImportExecutable",
        poptorch::processTraceAndImportExecutable);
  m.def("execute", poptorch::execute);
  m.def("getTimestamps", poptorch::getTimestamps);
  m.def("readOptimizerState", poptorch::readOptimizerState);
  m.def("loadEngineAndConnectStreams", poptorch::loadEngineAndConnectStreams);
  m.def("copyWeightsToDevice_impl", poptorch::copyWeightsToDeviceImpl);
  m.def("copyWeightsToHost_impl", poptorch::copyWeightsToHostImpl);
  m.def("ipuHardwareVersion", poptorch::ipuHardwareVersion,
        py::arg("numIpus") = 1);
  m.def("setLogLevel", poptorch::setLogLevel, py::arg("level") = 2);
  m.def("setPopartLogLevel", poptorch::setPopartLogLevelUInt);
  m.def("_getPopartIR", poptorch::getPopartIR);
  m.def("detachFromDevice", poptorch::detachFromDevice);
  m.def("attachToDevice", poptorch::attachToDevice);
  m.def("isAttachedToDevice", poptorch::isAttachedToDevice);
  m.def("registerCPUCallBack", poptorch::registerCPUCallBack);
  m.def("isAlreadyRegistered", poptorch::alreadyRegistered);
  m.def("registerBuffersWithCallback", poptorch::registerBuffersWithCallback);

  py::enum_<poptorch::TracingMode>(m, "TracingMode")
      .value("PopART", poptorch::TracingMode::POPART)
      .value("MLIR", poptorch::TracingMode::MLIR)
      .value("CPU", poptorch::TracingMode::CPU)
      .value("Sentinel", poptorch::TracingMode::SENTINEL)
      .export_values();

  m.def("startDispatch", poptorch::startDispatch);
  m.def("endDispatch", poptorch::endDispatch);
  m.def("createGraph", poptorch::createGraph);
  m.def("markOutputs", poptorch::markOutputs);
  m.def("compileWithManualTracing", poptorch::compileWithManualTracing);

  m.def("_throwTestError", poptorch::throwTestErrorSafe);

  static Error error(m, "Error");
  static RecoverableError recoverable_error(m, "RecoverableError", error);
  static Error unrecoverable_error(m, "UnrecoverableError", error);

  py::enum_<poptorch::TestErrorType>(m, "TestErrorType")
      .value("Poptorch", poptorch::TestErrorType::Poptorch)
      .value("Popart", poptorch::TestErrorType::Popart)
      .value("PopartInternal", poptorch::TestErrorType::PopartInternal)
      .value("Poplibs", poptorch::TestErrorType::Poplibs)
      .value("PoplarUnrecoverable",
             poptorch::TestErrorType::PoplarUnrecoverable)
      .value("PoplarUnknown", poptorch::TestErrorType::PoplarUnknown)
      .value("PoplarRecoverableFullReset",
             poptorch::TestErrorType::PoplarRecoverableFullReset)
      .value("PoplarLinkError", poptorch::TestErrorType::PoplarLinkError);

  py::register_exception_translator(
      [](std::exception_ptr p) { // NOLINT: Don't change 'p' to a const&
        try {
          if (p) {
            std::rethrow_exception(p);
          }
        } catch (const poptorch::ExceptionInfo &e) {
          for (int64_t i = e.stackDepth() - 1; i >= 0; --i) {
            poptorch::logging::LogContext::push(e.stack(i));
          }
          std::string location{};
          auto ctx = poptorch::logging::LogContext::context();
          if (ctx) {
            location = ctx.get();
          }
          std::stringstream what;
          what << "In " << e.filename() << ":" << e.line() << ": '" << e.type()
               << "': " << e.what();
          Error *err = nullptr;
          switch (e.category()) {
          case poptorch::ErrorCategory::RuntimeRecoverable: {
            what << "\nRecovery action required: " << e.recoveryAction();
            recoverable_error.setRecoveryAction(e.recoveryAction());
            err = &recoverable_error;
            break;
          }
          case poptorch::ErrorCategory::RuntimeUnrecoverable: {
            err = &unrecoverable_error;
            break;
          }
          default: {
            err = &error;
            break;
          }
          }

          if (!location.empty()) {
            what << "\nError raised in:\n" << location;
          }
          poptorch::logging::LogContext::resetContext();
          err->setType(e.type());
          err->setMessage(e.what());
          err->setLocation(location);
          // Note: on Ubuntu 20.04 PyErr_SetString(), i.e setWhat(),
          // needs to be the last call in register_exception_translator()
          err->setWhat(what.str());
        }
      });
}
