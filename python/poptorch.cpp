// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/script.h>

#include <iostream>
#include <unordered_map>

#include "popart_compiler/Compiler.hpp"

// Shared enums across the ABI boundary.
#include "popart_compiler/PopartEnums.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/EliminateListConstructs.hpp"
#include "poptorch/LowerToPopart.hpp"
#include "poptorch/Peephole.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/ShapeInference.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"

void beginIpuBlock(int64_t ipu_id) { UNUSED(ipu_id); }
void endIpuBlock() {}

at::Tensor identityOp(at::Tensor t) { return t; }
at::Tensor identityLoss(at::Tensor t, int64_t reduction) {
  UNUSED(reduction);
  return t;
}

c10::List<at::Tensor>
customOperation(c10::List<at::Tensor> inputs,            // NOLINT
                std::string name, std::string domain,    // NOLINT
                int64_t version, int64_t num_outputs,    // NOLINT
                c10::List<at::Tensor> example_outputs) { // NOLINT
  UNUSED(inputs);
  UNUSED(name);
  UNUSED(domain);
  UNUSED(version);
  UNUSED(num_outputs);

  return example_outputs;
}

static auto registry =
    torch::RegisterOperators("poptorch::begin_ipu_block", &beginIpuBlock)
        .op("poptorch::end_ipu_block", &endIpuBlock)
        .op("poptorch::ipu_print_tensor", &identityOp)
        .op("poptorch::nop", &identityOp)
        .op("poptorch::custom_operation", &customOperation)
        .op("poptorch::identity_loss", &identityLoss);
//.op("popart::convolution", convolution,
// torch::RegisterOperators::options().aliasAnalysis(c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE));

namespace poptorch {
namespace {

// Process the user provided dictionary and extract the relevant optimizer
// information.
Optimizer parseOptimizer(const py::dict &opt) {
  // optimizer is the map containing all set options.
  std::unordered_map<std::string, std::pair<float, bool>> optimizer;
  OptimizerType type = OptimizerType::NONE;

  // Extract all options from the python dictionary.
  for (auto element : opt) {
    // All values are in the form of pair{float, bool} except for the optimizer
    // option.
    if (py::isinstance<py::int_>(element.second)) {
      // Get the optimizer type.
      type = static_cast<OptimizerType>(element.second.cast<std::uint64_t>());
    } else {
      // Get the name of the optimizer option.
      const std::string name = element.first.cast<std::string>();
      optimizer[name] = element.second.cast<std::pair<float, bool>>();
    }
  }

  return Optimizer{type, optimizer};
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

// python_names and python_tensors are the parameters from the python trace.
// And trace_tensors is a subset of python_tensors (The unused parameters have
// been removed). So we build a map[tensor] = name based on the python trace
// which we then use to build the list of the names of the parameters in
// traced_tensors.
std::vector<std::string>
getParameterNames(const pybind11::tuple &python_names,
                  const pybind11::tuple &python_tensors,
                  const std::vector<at::Tensor> &traced_tensors) {
  std::vector<std::string> names;
  for (auto name : python_names) {
    ERROR_ON(!py::isinstance<py::str>(name));
    names.push_back(name.cast<std::string>());
  }

  torch::jit::Stack parameter_stack =
      torch::jit::toTraceableStack(python_tensors);
  std::map<void *, std::string> parameters_map;
  int name_idx = 0;
  for (const torch::jit::IValue &value : parameter_stack) {
    parameters_map.insert({value.toTensor().data_ptr(), names.at(name_idx)});
    name_idx++;
  }

  std::vector<std::string> parameters;
  parameters.reserve(traced_tensors.size());
  for (auto &param : traced_tensors) {
    parameters.push_back(parameters_map.at(param.data_ptr()));
  }
  return parameters;
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

SessionOptions parseSessionOptions(const py::dict &opt) {
  // steps, replicationFactor, profile
  SessionOptions options;

  for (auto element : opt) {
    // Exception _trace_model is only used by Python
    if (element.first.cast<std::string>() == "_trace_model") {
      continue;
    }
    if (py::isinstance<py::bool_>(element.second)) {
      options.addBoolOption(element.first.cast<std::string>().c_str(),
                            element.second.cast<bool>());
    } else if (py::isinstance<py::float_>(element.second)) {
      options.addDoubleOption(element.first.cast<std::string>().c_str(),
                              element.second.cast<double>());
    } else if (py::isinstance<py::int_>(element.second)) {
      options.addUint64Option(element.first.cast<std::string>().c_str(),
                              element.second.cast<std::uint64_t>());
    } else if (py::isinstance<py::str>(element.second)) {
      options.addStringOption(element.first.cast<std::string>().c_str(),
                              element.second.cast<std::string>().c_str());
    } else if (py::isinstance<py::set>(element.second) ||
               py::isinstance<py::list>(element.second)) {
      for (auto option : element.second.cast<py::list>()) {
        options.insertStringOption(element.first.cast<std::string>().c_str(),
                                   castToString(option).c_str());
      }
    } else if (py::isinstance<py::dict>(element.second)) {
      for (auto option : element.second.cast<py::dict>()) {
        options.insertStringPairOption(
            element.first.cast<std::string>().c_str(),
            option.first.cast<std::string>().c_str(),
            castToString(option.second).c_str());
      }
    } else {
      ERROR("Unknown option type " << element.second.get_type());
    }
  }
  return options;
}

void buildTensorList(const torch::jit::IValue &value,
                     std::vector<at::Tensor> *tensors) {
  if (value.isTuple()) {
    for (auto &element : value.toTuple()->elements()) {
      buildTensorList(element, tensors);
    }
  } else if (value.isList()) {
    for (const auto &element : value.toList()) {
      buildTensorList(element, tensors);
    }
  } else if (value.isTensor()) {
    tensors->push_back(value.toTensor());
  } else {
    ERROR("Unsupported value " << value.tagKind());
  }
}

} // namespace

void copyWeightsToHostImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors) {
  // Copy the weights or warn if this is before first time compilation.
  if (!executable) {
    logging::warn(
        "Call to copyWeightsToHost ignored as model has not been compiled "
        "(Poptorch will compile models on first invocation).");
  } else {
    executable->copyWeightsToHost(
        getParameterBuffers(parameter_names, parameter_tensors));
  }
}

void copyWeightsToDeviceImpl(
    const std::shared_ptr<poptorch::PoplarExecutable> &executable,
    const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors) {
  // Copy the weights or warn if this is before first time compilation.
  if (!executable) {
    logging::warn(
        "Call to copyWeightsToDevice ignored as model has not been compiled "
        "(Poptorch will compile models on first invocation).");
  } else {
    executable->copyWeightsToDevice(
        getParameterBuffers(parameter_names, parameter_tensors));
  }
}

std::string
getPopartIR(const std::shared_ptr<poptorch::PoplarExecutable> &executable) {
  return executable->getPopartIR();
}

std::vector<pybind11::object>
execute(const std::shared_ptr<poptorch::PoplarExecutable> &executable,
        const pybind11::tuple &inputs, py::dict *optimizerDict) {
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
    Optimizer optimizer{OptimizerType::NONE, {}};

    if (optimizerDict) {
      optimizer = parseOptimizer(*optimizerDict);
    }

    std::vector<at::IValue> output_tensors =
        executable->run(&input_tensors, optimizer);

    std::vector<pybind11::object> returnee;

    // Reshape the output tensors in the structure expected by the user
    auto tensor_it = output_tensors.begin();
    auto &output_types = executable->outputTypes();
    auto type_it = output_types.begin();
    ERROR_ON(type_it == output_types.end());
    std::uint64_t num_outputs = type_it->num_elements;
    std::function<pybind11::object()> process_output;
    process_output = [&]() -> pybind11::object { // NOLINT
      ERROR_ON_MSG(type_it == output_types.end(), "Invalid OutputTypes object");
      switch (type_it->type) {
      case OutputType::Type::Tensor: {
        ERROR_ON_MSG(tensor_it == output_tensors.end(),
                     "Not enough tensors to unpack");
        auto object = torch::jit::toPyObject(*tensor_it);
        tensor_it++;
        return object;
      }
      case OutputType::Type::Tuple: {
        std::int64_t num_elements = type_it->num_elements;
        pybind11::tuple pytuple(num_elements);
        for (std::int64_t i = 0; i < num_elements; ++i) {
          type_it++;
          pytuple[i] = process_output();
        }
        return std::move(pytuple);
      }
      case OutputType::Type::List: {
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

torch::jit::script::Module *asModule(py::handle h) {
  return reinterpret_cast<torch::jit::script::Module *>(
      pybind11::detail::values_and_holders(
          reinterpret_cast<pybind11::detail::instance *>(h.ptr()))
          .begin()
          ->value_ptr());
}

torch::jit::Graph *asGraph(py::handle h) {
  return reinterpret_cast<torch::jit::Graph *>(
      pybind11::detail::values_and_holders(
          reinterpret_cast<pybind11::detail::instance *>(h.ptr()))
          .begin()
          ->value_ptr());
}

void constantPropagation(torch::jit::Graph *graph) {
  // Create a shared_ptr with a custom deleter that doesn't do anything.
  std::shared_ptr<torch::jit::Graph> x(graph,
                                       [](torch::jit::Graph * /*unused*/) {});
  ERROR_ON_MSG(x.use_count() != 1, "x should be the only handle to graph");
  torch::jit::ConstantPropagation(x);
  ERROR_ON_MSG(x.use_count() != 1, "x should be the only handle to graph");
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithTrace(py::handle h, const pybind11::tuple &parameter_names,
                 const pybind11::tuple &parameter_tensors,
                 const pybind11::tuple &inputs, const pybind11::dict &options,
                 bool training, const py::dict &optimizerDict) {
  try {
    auto module = asModule(h);

    auto forward = module->get_method("forward");
    auto graph_and_tensors =
        torch::jit::LowerGraph(*forward.graph(), module->_ivalue());
    auto graph = graph_and_tensors.first;

    // Create a jit stack from the incoming pytorch tensors.
    torch::jit::Stack input_stack = torch::jit::toTraceableStack(inputs);

    // And turn convert them into at tensors which we can then resolve the
    // address of.
    std::vector<at::Tensor> input_tensors;
    for (const torch::jit::IValue &value : input_stack) {
      buildTensorList(value, &input_tensors);
    }

    std::vector<std::string> parameters = getParameterNames(
        parameter_names, parameter_tensors, graph_and_tensors.second);

    Optimizer optimizer = parseOptimizer(optimizerDict);

    logging::trace("Lowered graph:\n{}", *graph);

    torch::jit::EliminateDeadCode(graph);
    torch::jit::PeepholeOptimize(graph);
    torch::jit::EliminateDeadCode(graph);

    torch::jit::RemoveInplaceOps(graph);
    torch::jit::LowerSimpleTuples(graph);
    torch::jit::PeepholeOptimize(graph);

    torch::jit::RemoveInplaceOps(graph);

    logging::trace("Graph right before casting making integerer params as "
                   "constants inputs:\n{}",
                   *graph);
    poptorch::type_and_constant_canonicalization::makeConstantIntParams(
        graph.get(), parameters, graph_and_tensors.second);

    logging::trace("Graph right before casting unsupported inputs:\n{}",
                   *graph);
    poptorch::type_and_constant_canonicalization::castUnsupportedInputs(
        graph.get());

    logging::trace("Graph right before output type changes:\n{}", *graph);
    poptorch::type_and_constant_canonicalization::checkAndChangeOutputTypes(
        graph.get());

    logging::trace("Graph right before constant canonicalisation\n{}", *graph);
    poptorch::type_and_constant_canonicalization::canonicaliseConstants(
        graph.get());

    // Convert the IR to half to match the inputs/actual usage.
    logging::debug("Graph before canonicalising half:\n{}", *graph);
    poptorch::canonicaliseHalf(graph.get(), input_tensors,
                               graph_and_tensors.second);

    logging::debug("Graph right before canonicalization:\n{}", *graph);

    poptorch::canonicalizeLists(graph.get());

    // Convert any unsupported ATEN nodes in the graph to a popart
    // representation.
    poptorch::canonicalize(graph.get());

    // Enforce any constraints that aren't enforced by popart.
    poptorch::canonicalizeLate(graph.get());

    logging::debug("Graph right after canonicalization:\n{}", *graph);

    // Warn the user if any operations couldn't be canonicalised.
    poptorch::warnOnUnsupportedAten(graph.get());

    logging::debug("Graph right before popart:\n{}", *graph);

    return poptorch::lowerToPopart(graph.get(), &input_tensors,
                                   std::move(graph_and_tensors.second),
                                   std::move(parameters), training, optimizer,
                                   parseSessionOptions(options));
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

std::shared_ptr<poptorch::PoplarExecutable> compileWithScript(
    py::handle h, py::handle g, const pybind11::tuple &parameter_names,
    const pybind11::tuple &parameter_tensors, const pybind11::tuple &inputs,
    const pybind11::dict &options, bool training) {
  try {
    auto module = asModule(h);
    auto arg_graph = asGraph(g);

    torch::jit::Inline(*arg_graph);
    constantPropagation(arg_graph);
    peepholeOptimizations(arg_graph, training);

    auto graph_and_tensors =
        torch::jit::LowerGraph(*arg_graph, module->_ivalue());
    auto graph = graph_and_tensors.first;
    graph->dump();
    std::vector<std::string> parameters = getParameterNames(
        parameter_names, parameter_tensors, graph_and_tensors.second);

    int loop_count = 0;
    std::string graph_string;
    while (true) {
      propagateInputShapes(graph.get());
      torch::jit::PeepholeOptimize(graph, true);
      torch::jit::ConstantPropagation(graph);
      torch::jit::EliminateDeadCode(graph);
      peepholeOptimizations(graph.get(), training);

      std::string post_passes_graph = graph->toString(false);
      if (graph_string == post_passes_graph) {
        std::cout << "Breaking from const folding after " << loop_count
                  << " iterations.\n";
        break;
      }
      graph_string = std::move(post_passes_graph);

      loop_count++;
    }

    torch::jit::RemoveInplaceOps(graph);

    logging::trace("Graph right before casting unsupported inputs:\n{}",
                   *graph);
    poptorch::type_and_constant_canonicalization::castUnsupportedInputs(
        graph.get());

    logging::trace("Graph right before output type changes:\n{}", *graph);
    poptorch::type_and_constant_canonicalization::checkAndChangeOutputTypes(
        graph.get());

    logging::trace("Graph right before number constant replacement:\n{}",
                   *graph);
    poptorch::type_and_constant_canonicalization::canonicaliseConstants(
        graph.get());

    logging::debug("Graph right before canonicalization:\n{}", *graph);

    // Convert any unsupported ATEN nodes in the graph to a popart
    // representation.
    poptorch::canonicalize(graph.get());

    // Clean up the module as we will likely have stopped using lots of
    // constants.

    // Create a jit stack from the incoming pytorch tensors.
    torch::jit::Stack input_stack = torch::jit::toTraceableStack(inputs);

    // And turn convert them into at tensors which we can then resolve the
    // address of.
    std::vector<at::Tensor> input_tensors;
    for (const torch::jit::IValue &value : input_stack) {
      input_tensors.push_back(value.toTensor());
    }

    // Find the parameter data from.
    std::vector<at::Tensor> parameter_data = graph_and_tensors.second;
    std::cout << "There should be " << parameter_data.size()
              << " parameters.\n";

    logging::debug("Graph right before popart:\n{}", *graph);

    Optimizer optimizer{OptimizerType::NONE, {}};
    return poptorch::lowerToPopart(graph.get(), &input_tensors,
                                   std::move(parameter_data),
                                   std::move(parameters), training, optimizer,
                                   parseSessionOptions(options));
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void pyPropagateInputShapes(py::handle h) {
  try {
    auto graph = asGraph(h);
    propagateInputShapes(graph);
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void pyPeepholeOptimizations(py::handle h, bool training) {
  try {
    auto graph = asGraph(h);
    peepholeOptimizations(graph, training);
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void pyEliminateListConstructs(py::handle h) {
  try {
    auto graph = asGraph(h);
    eliminateListConstructs(graph);
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void pyCanonicalize(py::handle h) {
  try {
    auto graph = asGraph(h);
    canonicalize(graph);
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

} // namespace poptorch

PYBIND11_MODULE(poptorch_core, m) { // NOLINT
  py::class_<poptorch::PoplarExecutable,
             std::shared_ptr<poptorch::PoplarExecutable>>
      give_me_a_name(m, "InternalPoplarExecutable");

  m.def("compileWithTrace", poptorch::compileWithTrace);
  m.def("compileWithScript", poptorch::compileWithScript);
  m.def("execute", poptorch::execute);
  m.def("propagateInputShapes", poptorch::pyPropagateInputShapes);
  m.def("peepholeOptimizations", poptorch::pyPeepholeOptimizations);
  m.def("eliminateListConstructs", poptorch::pyEliminateListConstructs);
  m.def("canonicalize", poptorch::pyCanonicalize);
  m.def("copyWeightsToDevice_impl", poptorch::copyWeightsToDeviceImpl);
  m.def("copyWeightsToHost_impl", poptorch::copyWeightsToHostImpl);
  m.def("ipuHardwareIsAvailable", poptorch::ipuHardwareIsAvailable,
        py::arg("numIpus") = 1);
  m.def("_getPopartIR", poptorch::getPopartIR);
}
