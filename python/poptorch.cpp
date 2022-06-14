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

#include "popart_compiler/CodeletsCompilation.hpp"
#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/Utils.hpp"
// Shared enums across the ABI boundary.
#include "popart_compiler/PopartEnums.hpp"

#include "poptorch_err/ExceptionHandling.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "poptorch_logging/Tracepoint.hpp"

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

void beginIpuBlock(int64_t stage_id, int64_t phase_id, int64_t ipu_id) {
  UNUSED(stage_id);
  UNUSED(phase_id);
  UNUSED(ipu_id);
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
        .op("poptorch::nop", &identityOp)
        .op("poptorch::custom_operation", &customOperation)
        .op("poptorch::ctc_beam_search_decoder", &ctcBeamSearchDecoder)
        .op("poptorch::identity_loss", &identityLoss)
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

bool mlirIsSupportedOnPlatform() { return POPTORCH_BUILD_MLIR_COMPILER == 1; }

void registerBuffersWithCallback(
    const std::string &ID,
    std::vector<at::Tensor> &input_tensors, // NOLINT
    std::vector<at::Tensor> &output_tensors // NOLINT
) {
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
  CallbackMetadata &metadata = it->second;

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
std::vector<Optimizer> parseOptimizers(const py::dict &opt) {
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

SessionOptions parseSessionOptions(const py::dict &opts) {
  logging::LogContext ctx_func("parseSessionOptions");
  // steps, replicationFactor, profile
  SessionOptions options;

  for (const auto &element : opts) {
    const auto name = element.first.cast<std::string>();
    const auto value = element.second;
    logging::LogContext ctx("option: " + name);

    // Exception: patterns_level is handled at the same time as "patterns".
    // Exception: anchored_tensors is dealt with exclusively in Python.
    if (name == "patterns_level" || name == "anchored_tensors") {
      continue;
    }

    if (name == "compilation_progress_bar_fn") {
      options.setCompilationProgressLogger(value.cast<py::function>());
    } else if (py::isinstance<py::bool_>(value)) {
      options.addBoolOption(name.c_str(), value.cast<bool>());
    } else if (py::isinstance<py::float_>(value)) {
      options.addDoubleOption(name.c_str(), value.cast<double>());
    } else if (py::isinstance<py::int_>(value)) {
      options.addUint64Option(name.c_str(), value.cast<std::uint64_t>());
    } else if (py::isinstance<py::str>(value)) {
      options.addStringOption(name.c_str(), value.cast<std::string>().c_str());
    } else if (py::isinstance<py::set>(value) ||
               py::isinstance<py::list>(value)) {
      for (auto option : value.cast<py::list>()) {
        options.insertStringOption(name.c_str(), castToString(option).c_str());
      }
    } else if (py::isinstance<py::dict>(value)) {
      if (name == "available_memory_proportion") {
        for (auto option : value.cast<py::dict>()) {
          options.setMemoryProportion(option.first.cast<std::uint64_t>(),
                                      option.second.cast<float>());
        }
      } else if (name == "patterns") {
        ERROR_ON_MSG(!opts.contains("patterns_level"),
                     "PopART option 'patterns' should not be set "
                     "without first setting 'patterns_level'.");

        options.setPatternsLevel(opts["patterns_level"].cast<std::uint64_t>());

        for (auto option : value.cast<py::dict>()) {
          options.addPattern(option.first.cast<std::string>().c_str(),
                             option.second.cast<bool>());
        }
      } else if (name.rfind("location_", 0) == 0) {
        for (auto option : value.cast<py::dict>()) {
          options.setTensorLocation(name.c_str(),
                                    option.first.cast<std::string>().c_str(),
                                    option.second.cast<std::uint64_t>());
        }
      } else {
        for (auto option : value.cast<py::dict>()) {
          options.insertStringPairOption(
              name.c_str(), option.first.cast<std::string>().c_str(),
              castToString(option.second).c_str());
        }
      }
    } else {
      ERROR("Unknown value type " << py::str(value.get_type()) << " for option "
                                  << name);
    }
  }

  return options;
}

void parseSessionOptionsVoid(const py::dict &opts) {
  parseSessionOptions(opts);
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
  const char *poptorch_passes = "poptorchPasses";
  const char *lower_to_popart = "lowerToPopart";
  poptorch::logging::Tracepoint::begin(poptorch_passes);
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

  std::vector<Optimizer> optimizers = parseOptimizers(optimizer_dict);
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

  InplaceGraphInfo inplace_info = handleInplaceOpsInGraph(
      *graph, traced_parameter_tensors.size(), anchors.size(),
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

  poptorch::removeScatterAddIndexExpansion(graph.get());

  poptorch::simplifyGatherWithExpandedIndices(graph.get());

  logging::trace("Graph right before canonicalization:\n{}", *graph);
  // Convert any unsupported ATEN nodes in the graph to a popart
  // representation.
  poptorch::canonicalize(graph.get());

  printGraphBeforeHalfFloatResolution(*graph);

  poptorch::annotateSubgraphs(graph.get(), graph->nodes().front());

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

  poptorch::logging::Tracepoint::end(poptorch_passes);
  poptorch::logging::Tracepoint::begin(lower_to_popart);

  poptorch::LowerToPopart lower(
      graph.get(), traced_parameter_tensors, parameters,
      std::move(inplace_info), training, std::move(optimizers), parsed_options,
      attribute_accessor, callbacks, std::move(anchors_list));
  lower.lower(&input_tensors);

  // Clear the callbacks after compilation.
  callbacks.clear();

  poptorch::setAvailableMemoryOnGraphFinalized();

  poptorch::logging::Tracepoint::end(lower_to_popart);
  return lower;
}

void mapParamsToNames(const pybind11::tuple &names,
                      const pybind11::tuple &tensors) {
  ERROR_ON(names.size() != tensors.size());
  torch::jit::Stack stack = torch::jit::toTraceableStack(tensors);
  for (uint64_t i = 0; i < names.size(); ++i) {
    const auto name = names[i].cast<std::string>();
    const auto tensor = stack[i].toTensor();
    setParameterName(tensor, name);
  }
}
} // namespace

void copyWeightsToHostImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors) {
  poptorch::logging::Tracepoint tp{"copyWeightsToHost"};
  // Copy the weights or warn if this is before first time compilation.
  if (!executable) {
    logging::warn(
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
    logging::warn(
        "Call to copyWeightsToDevice ignored as model has not been compiled "
        "(PopTorch will compile models on first invocation).");
  } else {
    executable->copyWeightsToDevice(
        getParameterBuffers(parameter_names, parameter_tensors));
  }
}

std::string
getPopartIR(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  ERROR_ON_MSG(!executable, "No built executable");
  return executable->getPopartIR();
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

void setPopartLogLevelUInt(std::uint64_t level) {
  ERROR_ON(level > static_cast<std::uint64_t>(logging::Level::Off) ||
           level == 5);
  poptorch::setPopartLogLevel(static_cast<logging::Level>(level));
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
  std::vector<Optimizer> optimizers = parseOptimizers(optimizer_dict);

  executable->updateOptimizers(optimizers);
}

std::vector<pybind11::object>
execute(const std::shared_ptr<poptorch::PoplarExecutable> &executable,
        const pybind11::tuple &inputs) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  ERROR_ON_MSG(!executable, "No built executable");
  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack input_stack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> input_tensors;
  for (const torch::jit::IValue &value : input_stack) {
    buildTensorList(value, &input_tensors);
  }

  std::vector<at::IValue> output_tensors = executable->run(&input_tensors);

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
  std::vector<TensorMetadata> metadata_list =
      compiler.optimizerTensorMetadataList();

  std::vector<void *> host_buffers;
  auto state = optim_state["ipu_state"];
  auto params = optim_state["ipu_param"];

  for (const TensorMetadata &meta : metadata_list) {
    if (meta.num_bytes == -1) {
      // num_bytes == -1 indicates it's an optimiser state tensor (variable)
      if (!state.contains(py::cast(meta.id))) {
        logging::warn("writeOptimizerState: ignoring missing state {}",
                      meta.id);
        host_buffers.push_back(nullptr);
        continue;
      }
      at::Tensor tensor = state[py::cast(meta.id)].cast<at::Tensor>();
      host_buffers.push_back(tensor.data_ptr());
    } else {
      if (!params.contains(py::cast(meta.id))) {
        logging::warn("writeOptimizerState: ignoring missing parameter {}",
                      meta.id);
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
  auto num_inputs = compiler.getNumInputs();
  auto num_outputs = compiler.getNumOutputs();

  py::list input;
  py::list input_complete;
  py::list output;
  py::list output_complete;

  for (size_t i = 0; i < num_inputs; ++i) {
    input.append(py::cast(compiler.getInputTimestamps(i)));
    input_complete.append(py::cast(compiler.getInputCompleteTimestamps(i)));
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    output.append(py::cast(compiler.getOutputTimestamps(i)));
    output_complete.append(py::cast(compiler.getOutputCompleteTimestamps(i)));
  }

  return {input, input_complete, output, output_complete};
}

void processPrecisionOptions(py::handle h) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
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
  Compiler::appendPoptorchMetadataToFile(serialized_poptorch_data.c_str(),
                                         serialized_poptorch_data.size(),
                                         export_filename.c_str());
}

uint64_t
cycleCount(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  ERROR_ON_MSG(!executable, "No built executable");
  return executable->getCompiler().getCycleCount();
}

std::shared_ptr<poptorch::PoplarExecutable> processTraceAndImportExecutable(
    py::handle h, const pybind11::dict &python_traced_params,
    const pybind11::tuple &inputs, bool has_converted_any_half,
    const pybind11::dict &options, bool training,
    const py::dict &optimizer_dict, const py::function &attribute_accessor,
    const bool added_dummy_output, const py::list &anchors,
    const py::dict &model_parameters, const std::string &import_filename) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  auto lower = lowerToPopartFromTrace(
      h, python_traced_params, inputs, has_converted_any_half, options,
      training, optimizer_dict, attribute_accessor, added_dummy_output, anchors,
      model_parameters);
  return lower.loadExecutableFromFile(import_filename);
}

py::bytes importPoptorchMetadataFromFile(const std::string &import_filename) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  std::vector<char> metadata_buffer =
      Compiler::importPoptorchMetadataFromFile(import_filename.c_str());
  return py::bytes(metadata_buffer.data(), metadata_buffer.size());
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithTrace(py::handle h, const pybind11::dict &python_traced_params,
                 const pybind11::tuple &inputs, bool has_converted_any_half,
                 const pybind11::dict &options, bool training,
                 const py::dict &optimizer_dict,
                 const py::function &attribute_accessor,
                 const bool added_dummy_output, const py::list &anchors,
                 const py::dict &model_parameters) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  auto lower = lowerToPopartFromTrace(
      h, python_traced_params, inputs, has_converted_any_half, options,
      training, optimizer_dict, attribute_accessor, added_dummy_output, anchors,
      model_parameters);
  return lower.compile();
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithManualTracing(const pybind11::dict &options,
                         const py::function &attribute_accessor,
                         bool is_training, const py::dict &opt_dict) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  logging::debug("Compile with manual tracing");

  SessionOptions parsed_options = parseSessionOptions(options);

  AnchorList anchors_list;
  std::vector<Optimizer> optimizers = parseOptimizers(opt_dict);

  InplaceGraphInfo inplace_info = getInplaceGraphInfo(
      anchors_list.size(), parsed_options.replicationFactor() > 1 &&
                               parsed_options.broadcastBuffers());
  std::shared_ptr<torch::jit::Graph> graph = getTracedGraph();

  logging::debug("Traced graph:\n{}", *graph);

  // Make sure all constants are correctly categorised as either
  // poptorch::tensor_constant or poptorch::host_side_tensor_constant now
  // that we have a full graph.
  poptorch::type_and_constant_canonicalization::categoriseConstantsDispatch(
      graph.get());

  if (graph->outputs().empty()) {
    logging::trace("No outputs, so all nodes cleared");
    for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend(); it++) {
      it.destroyCurrent();
    }
  }

  // TODO(T55228): remove after we use our own dispatch key.
  removeDeadImplicitCasts(graph.get());

  poptorch::LowerToPopart lower(
      graph.get(), std::move(inplace_info), is_training, std::move(optimizers),
      parsed_options, attribute_accessor, callbacks, std::move(anchors_list));

  lower.lower(nullptr);

  // We need to keep the dispatcher alive until after the passes because
  // some of them call isDispatcherActive() and until after the lowering
  // because the dispatcher is used to retrieve data pointers associated
  // with jit::Value for inputs and parameters.
  destroyDispatcher();

  return lower.compile();
}

} // namespace poptorch

PYBIND11_MODULE(poptorch_core, m) { // NOLINT
  py::class_<poptorch::PoplarExecutable,
             std::shared_ptr<poptorch::PoplarExecutable>>
      give_me_a_name(m, "InternalPoplarExecutable");

  m.def("processPrecisionOptions", PTC(poptorch::processPrecisionOptions));
  m.def("isGraphNondeterministic", PTC(poptorch::pyIsGraphNondeterministic));
  m.def("saveExecutableToFile", PTC(poptorch::saveExecutableToFile));
  m.def("appendPoptorchMetadataToFile",
        PTC(poptorch::appendPoptorchMetadataToFile));
  m.def("compileWithTrace", PTC(poptorch::compileWithTrace));
  m.def("cycleCount", PTC(poptorch::cycleCount));
  m.def("processTraceAndImportExecutable",
        PTC(poptorch::processTraceAndImportExecutable));
  m.def("importPoptorchMetadataFromFile",
        PTC(poptorch::importPoptorchMetadataFromFile));
  m.def("execute", PTC(poptorch::execute));
  m.def("updateOptimizers", PTC(poptorch::updateOptimizers));
  m.def("getTimestamps", PTC(poptorch::getTimestamps));
  m.def("readOptimizerState", PTC(poptorch::readOptimizerState));
  m.def("setRngState", PTC(poptorch::setRngState));
  m.def("getRngState", PTC(poptorch::getRngState));
  m.def("getRandomSeed", PTC(poptorch::getRandomSeed));
  m.def("writeOptimizerState", PTC(poptorch::writeOptimizerState));
  m.def("loadEngineAndConnectStreams",
        PTC(poptorch::loadEngineAndConnectStreams));
  m.def("copyWeightsToDevice_impl", PTC(poptorch::copyWeightsToDeviceImpl));
  m.def("copyWeightsToHost_impl", PTC(poptorch::copyWeightsToHostImpl));
  m.def("ipuHardwareVersion", PTC(poptorch::ipuHardwareVersion),
        py::arg("numIpus") = 1);
  m.def("setCustomCodeletsPath", PTC(poptorch::setCustomCodeletsPath));
  m.def("setLogLevel", PTC(poptorch::setLogLevel), py::arg("level") = 2);
  m.def("setPopartLogLevel", PTC(poptorch::setPopartLogLevelUInt));
  m.def("_getPopartIR", PTC(poptorch::getPopartIR));
  m.def("detachFromDevice", PTC(poptorch::detachFromDevice));
  m.def("attachToDevice", PTC(poptorch::attachToDevice));
  m.def("isAttachedToDevice", PTC(poptorch::isAttachedToDevice));
  m.def("registerCPUCallBack", PTC(poptorch::registerCPUCallBack));
  m.def("isAlreadyRegistered", PTC(poptorch::alreadyRegistered));
  m.def("registerBuffersWithCallback",
        PTC(poptorch::registerBuffersWithCallback));
  m.def("mlirIsSupportedOnPlatform", PTC(poptorch::mlirIsSupportedOnPlatform));
  m.def("_validateOptions", PTC(poptorch::parseSessionOptionsVoid));

#if POPTORCH_BUILD_MLIR_COMPILER
  py::class_<poptorch::MLIRExecutor, std::shared_ptr<poptorch::MLIRExecutor>>(
      m, "MLIRExecutor")
      .def("execute", &poptorch::MLIRExecutor::execute)
      .def("weightsToDevice", &poptorch::MLIRExecutor::weightsToDevice)
      .def("weightsToHost", &poptorch::MLIRExecutor::weightsToHost);
  m.def("compileWithMLIR", PTC(poptorch::compileMLIR));
#endif

  py::enum_<poptorch::TracingMode>(m, "TracingMode")
      .value("PopART", poptorch::TracingMode::POPART)
      .value("MLIR", poptorch::TracingMode::MLIR)
      .value("Sentinel", poptorch::TracingMode::SENTINEL)
      .export_values();

  m.def("enableEagerMode", PTC(poptorch::enableEagerMode));
  m.def("startDispatch", PTC(poptorch::startDispatch));
  m.def("endDispatch", PTC(poptorch::endDispatch));
  m.def("startParametersMove", PTC(poptorch::startParametersMove));
  m.def("endParametersMove", PTC(poptorch::endParametersMove));
  m.def("createGraph", PTC(poptorch::createGraph));
  m.def("mapParamsToNames", PTC(poptorch::mapParamsToNames));
  m.def("markOutputs", PTC(poptorch::markOutputs));
  m.def("finalizeGraph", PTC(poptorch::finalizeGraph));
  m.def("compileWithManualTracing", PTC(poptorch::compileWithManualTracing));
  m.def("_throwTestError", PTC(poptorch::throwTestError));

  poptorch::initialiseExceptionHandling(m);

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
        } catch (const poptorch::PoptorchError &e) {
          e.setErrorIndicator();
        }
      });
}
