// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/script.h>

#include <iostream>
#include <unordered_map>

#include "popart_compiler/Compiler.hpp"
#include "poptorch/EliminateListConstructs.hpp"
#include "poptorch/LowerToPopart.hpp"
#include "poptorch/Peephole.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/ShapeInference.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

// Shared enums across the ABI boundary.
#include "popart_compiler/PopartEnums.hpp"

void begin_ipu_block(int64_t ipu_id) { UNUSED(ipu_id); }
void end_ipu_block() {}

at::Tensor ipu_print_tensor(at::Tensor t) { return t; }
at::Tensor identity_loss(at::Tensor t, int64_t reduction) {
  UNUSED(reduction);
  return t;
}

static auto registry =
    torch::RegisterOperators("poptorch::begin_ipu_block", &begin_ipu_block)
        .op("poptorch::end_ipu_block", &end_ipu_block)
        .op("poptorch::ipu_print_tensor", &ipu_print_tensor)
        .op("poptorch::identity_loss", &identity_loss);
//.op("popart::convolution", convolution,
// torch::RegisterOperators::options().aliasAnalysis(c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE));

namespace poptorch {

namespace {

// Process the user provided dictionary and extract the relevant optimizer
// information.
Optimizer parseOptimizer(const py::dict &opt) {
  std::unordered_map<std::string, std::pair<float, bool>> optimizer;

  for (auto element : opt) {
    std::pair<float, bool> p = element.second.cast<std::pair<float, bool>>();

    optimizer[element.first.cast<std::string>()] = p;
  }

  return {optimizer};
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
      options.AddBoolOption(element.first.cast<std::string>().c_str(),
                            element.second.cast<bool>());
    } else if (py::isinstance<py::float_>(element.second)) {
      options.AddDoubleOption(element.first.cast<std::string>().c_str(),
                              element.second.cast<double>());
    } else if (py::isinstance<py::int_>(element.second)) {
      options.AddUInt64Option(element.first.cast<std::string>().c_str(),
                              element.second.cast<std::uint64_t>());
    } else if (py::isinstance<py::str>(element.second)) {
      options.AddStringOption(element.first.cast<std::string>().c_str(),
                              element.second.cast<std::string>().c_str());
    } else if (py::isinstance<py::set>(element.second) ||
               py::isinstance<py::list>(element.second)) {
      for (auto option : element.second.cast<py::list>()) {
        options.InsertStringOption(element.first.cast<std::string>().c_str(),
                                   castToString(option).c_str());
      }
    } else if (py::isinstance<py::dict>(element.second)) {
      for (auto option : element.second.cast<py::dict>()) {
        options.InsertStringPairOption(
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
                     std::vector<at::Tensor> &tensors) {
  if (value.isTuple()) {
    for (auto &element : value.toTuple()->elements()) {
      buildTensorList(element, tensors);
    }
  } else if (value.isList()) {
    for (auto element : value.toList()) {
      buildTensorList(element, tensors);
    }
  } else if (value.isTensor()) {
    tensors.push_back(value.toTensor());
  } else {
    ERROR("Unsupported value " << value.tagKind());
  }
}

} // namespace

void copyWeightsToHost_impl(
    std::shared_ptr<poptorch::PoplarExecutable> executable) {
  // Copy the weights or warn if this is before first time compilation.
  if (!executable) {
    logging::warn(
        "Call to copyWeightsToHost ignored as model has not been compiled "
        "(Poptorch will compile models on first invocation).");
  } else {
    executable->CopyWeightsToHost();
  }
}

void copyWeightsToDevice_impl(
    std::shared_ptr<poptorch::PoplarExecutable> executable) {
  // Copy the weights or warn if this is before first time compilation.
  if (!executable) {
    logging::warn(
        "Call to copyWeightsToDevice ignored as model has not been compiled "
        "(Poptorch will compile models on first invocation).");
  } else {
    executable->CopyWeightsToDevice();
  }
}

std::vector<pybind11::object>
execute(std::shared_ptr<poptorch::PoplarExecutable> executable,
        pybind11::tuple inputs, py::dict *optimizerDict) {
  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    buildTensorList(value, inputTensors);
  }

  Optimizer optimizer{{}};

  if (optimizerDict) {
    optimizer = parseOptimizer(*optimizerDict);
  }

  std::vector<at::IValue> outputTensors =
      executable->Run(inputTensors, optimizer);

  std::vector<pybind11::object> returnee;

  // Reshape the output tensors in the structure expected by the user
  auto tensorIt = outputTensors.begin();
  auto &outputTypes = executable->OutputTypes();
  auto typeIt = outputTypes.begin();
  ERROR_ON(typeIt == outputTypes.end());
  std::uint64_t numOutputs = typeIt->numElements;
  std::function<pybind11::object()> processOutput;
  processOutput = [&]() -> pybind11::object {
    ERROR_ON_MSG(typeIt == outputTypes.end(), "Invalid OutputTypes object");
    switch (typeIt->type) {
    case OutputType::Type::Tensor: {
      ERROR_ON_MSG(tensorIt == outputTensors.end(),
                   "Not enough tensors to unpack");
      auto object = torch::jit::toPyObject(*tensorIt);
      tensorIt++;
      return object;
    }
    case OutputType::Type::Tuple: {
      std::int64_t numElements = typeIt->numElements;
      pybind11::tuple pytuple(numElements);
      for (std::int64_t i = 0; i < numElements; ++i) {
        typeIt++;
        pytuple[i] = processOutput();
      }
      return std::move(pytuple);
    }
    case OutputType::Type::List: {
      std::int64_t numElements = typeIt->numElements;
      pybind11::list pylist(numElements);
      for (std::int64_t i = 0; i < numElements; ++i) {
        typeIt++;
        pylist[i] = processOutput();
      }
      return std::move(pylist);
    }
    default:
      ERROR("Unsupported OutputType");
    }
  };

  for (std::uint64_t i = 0; i < numOutputs; ++i) {
    typeIt++;
    returnee.push_back(processOutput());
  }
  ERROR_ON_MSG(tensorIt != outputTensors.end(),
               "Not all the output tensors were unpacked");

  return returnee;
}

torch::jit::script::Module *as_module(py::handle h) {
  return reinterpret_cast<torch::jit::script::Module *>(
      pybind11::detail::values_and_holders(
          reinterpret_cast<pybind11::detail::instance *>(h.ptr()))
          .begin()
          ->value_ptr());
}

torch::jit::Graph *as_graph(py::handle h) {
  return reinterpret_cast<torch::jit::Graph *>(
      pybind11::detail::values_and_holders(
          reinterpret_cast<pybind11::detail::instance *>(h.ptr()))
          .begin()
          ->value_ptr());
}

void constantPropagation(torch::jit::Graph *graph) {
  // Create a shared_ptr with a custom deleter that doesn't do anything.
  std::shared_ptr<torch::jit::Graph> x(graph, [](torch::jit::Graph *) {});
  ERROR_ON_MSG(x.use_count() != 1, "x should be the only handle to graph");
  torch::jit::ConstantPropagation(x);
  ERROR_ON_MSG(x.use_count() != 1, "x should be the only handle to graph");
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithTrace(py::handle h, pybind11::tuple inputs, pybind11::dict options,
                 bool training, py::dict optimizerDict) {
  auto module = as_module(h);

  auto forward = module->get_method("forward");
  auto graphAndTensors =
      torch::jit::LowerGraph(*forward.graph(), module->_ivalue());
  auto graph = graphAndTensors.first;

  Optimizer optimizer = parseOptimizer(optimizerDict);

  torch::jit::EliminateDeadCode(graph);
  torch::jit::PeepholeOptimize(graph);
  torch::jit::EliminateDeadCode(graph);

  torch::jit::RemoveInplaceOps(graph);

  logging::debug("Graph right before canonicalization:\n{}", *graph);

  poptorch::CanonicalizeLists(*graph);

  // Convert any unsupported ATEN nodes in the graph to a popart
  // representation.
  poptorch::Canonicalize(*graph);

  // Enforce any constraints that aren't enforced by popart.
  poptorch::CanonicalizeLate(*graph);

  // Warn the user if any operations couldn't be canonicalised.
  poptorch::WarnOnUnsupportedAten(*graph);

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    buildTensorList(value, inputTensors);
  }

  // Find the parameter data from.
  std::vector<at::Tensor> parameterData;
  for (at::Tensor param : graphAndTensors.second) {
    if (!param.is_contiguous()) {
      logging::debug("Tensor is NOT continguous!");
    }

    parameterData.push_back(param);
  }

  logging::debug("Graph right before popart:\n{}", *graph);

  return poptorch::lowerToPopart(*graph, inputTensors, parameterData, training,
                                 optimizer, parseSessionOptions(options));
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithScript(py::handle h, py::handle g, pybind11::tuple inputs,
                  pybind11::dict options, bool training) {
  auto module = as_module(h);
  auto argGraph = as_graph(g);

  torch::jit::Inline(*argGraph);
  constantPropagation(argGraph);
  peepholeOptimizations(*argGraph, training);

  auto graphAndTensors = torch::jit::LowerGraph(*argGraph, module->_ivalue());
  auto graph = graphAndTensors.first;
  graph->dump();

  int loop_count = 0;
  std::string graphString;
  while (true) {
    propagateInputShapes(graph.get());
    torch::jit::PeepholeOptimize(graph, true);
    torch::jit::ConstantPropagation(graph);
    torch::jit::EliminateDeadCode(graph);
    peepholeOptimizations(*graph, training);

    std::string postPassesGraph = graph->toString(false);
    if (graphString == postPassesGraph) {
      std::cout << "Breaking from const folding after " << loop_count
                << " iterations.\n";
      break;
    } else {
      graphString = std::move(postPassesGraph);
    }

    loop_count++;
  }

  torch::jit::RemoveInplaceOps(graph);

  logging::debug("Graph right before canonicalization:\n{}", *graph);

  // Convert any unsupported ATEN nodes in the graph to a popart
  // representation.
  poptorch::Canonicalize(*graph);

  // Clean up the module as we will likely have stopped using lots of
  // constants.

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    inputTensors.push_back(value.toTensor());
  }

  // Find the parameter data from.
  std::vector<at::Tensor> parameterData = graphAndTensors.second;
  std::cout << "There should be " << parameterData.size() << " parameters.\n";

  logging::debug("Graph right before popart:\n{}", *graph);

  Optimizer optimizer{{}};
  return poptorch::lowerToPopart(*graph, inputTensors, parameterData, training,
                                 optimizer, parseSessionOptions(options));
}

void pyPropagateInputShapes(py::handle h) {
  auto graph = as_graph(h);
  propagateInputShapes(graph);
}

void pyPeepholeOptimizations(py::handle h, bool training) {
  auto graph = as_graph(h);
  peepholeOptimizations(*graph, training);
}

void pyEliminateListConstructs(py::handle h) {
  auto graph = as_graph(h);
  eliminateListConstructs(graph);
}

void pyCanonicalize(py::handle h) {
  auto graph = as_graph(h);
  Canonicalize(*graph);
}

} // namespace poptorch

PYBIND11_MODULE(poptorch_core, m) {
  py::class_<poptorch::PoplarExecutable,
             std::shared_ptr<poptorch::PoplarExecutable>>(
      m, "InternalPoplarExecutable");

  m.def("compileWithTrace", poptorch::compileWithTrace);
  m.def("compileWithScript", poptorch::compileWithScript);
  m.def("execute", poptorch::execute);
  m.def("propagateInputShapes", poptorch::pyPropagateInputShapes);
  m.def("peepholeOptimizations", poptorch::pyPeepholeOptimizations);
  m.def("eliminateListConstructs", poptorch::pyEliminateListConstructs);
  m.def("canonicalize", poptorch::pyCanonicalize);
  m.def("copyWeightsToDevice_impl", poptorch::copyWeightsToDevice_impl);
  m.def("copyWeightsToHost_impl", poptorch::copyWeightsToHost_impl);
}
