// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/LowerToPopart.hpp"

#include <torch/csrc/jit/ir/ir.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <list>
#include <random>

#include "popart_compiler/Compiler.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

/*
 * Implementation of the lowering operation.
 */
class LowerToPopart {
public:
  LowerToPopart(torch::jit::Graph &g, std::vector<at::Tensor> &ins,
                std::vector<at::Tensor> &params, std::uint64_t steps,
                bool training, std::uint64_t replicationFactor,
                std::uint64_t gradientAccumulation, bool profile_);

  void Lower();

  std::shared_ptr<poptorch::PoplarExecutable> Compile();

private:
  torch::jit::Graph &graph;

  std::vector<at::Tensor> &inTensors;

  std::vector<at::Tensor> &parameters;

  std::vector<poptorch::TensorId> inputTensorHooks;

  std::vector<poptorch::TensorId> outputTensorHooks;

  // Mapping between the SSA values of torch jit with the ssa values of popart.
  std::unordered_map<torch::jit::Value *, std::vector<poptorch::TensorId>>
      valueMap;

  using FunctionType = std::function<poptorch::TensorId(
      const std::vector<poptorch::TensorId> &inputs, torch::jit::Node *)>;
  std::unordered_map<std::string, FunctionType> functionToImplementation;

  poptorch::Compiler compiler;

  bool profile;

  void LowerParameters();

  void LowerBody();

  void LowerReturn();
};

/*
 * Static helper functions.
 */

std::string typeToPopart(at::ScalarType type) {
  if (type == at::ScalarType::Float) {
    return "FLOAT";
  } else if (type == at::ScalarType::Int || type == at::ScalarType::Long) {
    return "INT32";
  }

  logging::err("Unimplemented type '{}'", type);
  return "UNIMPLEMENTED";
}

/*
 * Lower to popart impl.
 */
std::shared_ptr<poptorch::PoplarExecutable> LowerToPopart::Compile() {
  // Init the session, this also involves compiling to poplar.
  compiler.InitSession(profile);

  return std::make_shared<poptorch::PoplarExecutable>(
      std::move(compiler), std::move(inputTensorHooks),
      std::move(outputTensorHooks), profile);
}

void LowerToPopart::Lower() {
  // Lower the tensor parameters of the graph to OpInputs.
  LowerParameters();

  // Lower the body of the graph.
  LowerBody();

  LowerReturn();
}

void LowerToPopart::LowerReturn() {
  for (torch::jit::Value *value : graph.outputs()) {
    for (poptorch::TensorId id : valueMap[value]) {
      compiler.AddOutput(id);
      outputTensorHooks.push_back(id);
    }
  }
}

// Lower the main body of the graph.
void LowerToPopart::LowerBody() {
  for (torch::jit::Node *node : graph.nodes()) {
    // Switch/lookup based on the actual int value.
    const std::string &bodyAsStr = node->kind().toDisplayString();

    std::vector<poptorch::TensorId> inputs;
    std::transform(node->inputs().begin(), node->inputs().end(),
                   std::back_inserter(inputs), [&](torch::jit::Value *val) {
                     // Tuples aren't supported here but we don't support any
                     // operations which actually take in tuples.
                     return valueMap[val][0];
                   });

    auto itr = functionToImplementation.find(bodyAsStr);

    if (itr != functionToImplementation.end()) {
      // Get the torch jit SSA for the input/output values.
      torch::jit::Value *output = node->output();
      // Call the callback.
      valueMap[output].push_back(itr->second(inputs, node));
    } else if (bodyAsStr == "poptorch::begin_ipu_block") {
      compiler.SetActiveIpu(node->i(c10::Symbol::fromQualString("attr::ipu")));

    } else if (bodyAsStr == "poptorch::end_ipu_block") {
      // NOP for now.
    } else if (bodyAsStr == "prim::TupleConstruct" ||
               bodyAsStr == "prim::ListConstruct") {
      // Get the torch jit SSA for the input/output values.
      torch::jit::Value *output = node->output();

      // Add the values to the value map.
      for (torch::jit::Value *ids : node->inputs()) {
        for (poptorch::TensorId values : valueMap[ids]) {
          valueMap[output].push_back(values);
        }
      }
    } else if (bodyAsStr == "prim::TupleUnpack" ||
               bodyAsStr == "prim::ListUnpack") {
      // Get the torch jit SSA for the input/output values.
      at::ArrayRef<torch::jit::Value *> output = node->outputs();

      torch::jit::Value *input = node->input();
      // Mapping from a single tuple input which we record each tuple element as
      // being an output (not what is in the IR) to the actual unpack in the IR
      // which is just a pass through of what we've already done.
      for (std::int32_t i = 0; i < output.size(); ++i) {
        valueMap[output[i]].push_back(valueMap[input][i]);
      }
    } else {
      logging::err("Couldn't find a registered operation for node {}", *node);
    }
  }
}

void LowerToPopart::LowerParameters() {
  // Lower user provided inputs first.
  std::size_t index = 0;
  for (at::Tensor &tensor : inTensors) {
    // Convert the tensor type to the correct vector size.
    std::vector<int64_t> dims;
    std::transform(tensor.sizes().begin(), tensor.sizes().end(),
                   std::back_inserter(dims), [](std::int64_t i) { return i; });

    // Return the input tensor id for input tensor of given type and dims.
    poptorch::TensorId id = compiler.AddInputTensor(
        typeToPopart(tensor.scalar_type()).c_str(), dims);

    // Record the id so we can map back to the pytorch tensor.
    valueMap[graph.param_node()->outputs()[index]].push_back(id);
    inputTensorHooks.push_back(id);
    ++index;
  }

  // Then lower the other params (I.E the weights.)
  std::size_t paramIndex = 0;
  for (torch::jit::Value *value : graph.param_node()->outputs()) {
    // Skip the values already added (I.E)
    if (valueMap.find(value) != valueMap.end())
      continue;

    at::Tensor &tensorAsParam = parameters[paramIndex];

    // Convert the tensor type to the correct vector size.
    std::vector<int64_t> dims;
    std::transform(tensorAsParam.sizes().begin(), tensorAsParam.sizes().end(),
                   std::back_inserter(dims), [](std::int64_t i) { return i; });

    // Unpack the elem type into its Popart type.
    std::string popartType = typeToPopart(tensorAsParam.scalar_type());

    valueMap[value].push_back(compiler.AddInitializedInputTensor(
        "Weight", popartType.c_str(), dims, tensorAsParam.data_ptr()));

    paramIndex++;
  }
}

LowerToPopart::LowerToPopart(torch::jit::Graph &g, std::vector<at::Tensor> &ins,
                             std::vector<at::Tensor> &params,
                             std::uint64_t steps, bool training,
                             std::uint64_t replicationFactor,
                             std::uint64_t gradientAccumulation, bool profile_)
    : graph(g), inTensors(ins), parameters(params),
      compiler({training, steps, replicationFactor, gradientAccumulation}),
      profile(profile_) {
  // Init the function implementation map. This map will be populated by
  // elements which look something like:
  /* {"popart::Foo", [&](const std::vector<poptorch::TensorId> &inputs,
     torch::jit::Node *node) { return compiler.foo(inputs,
          node->i("attr::SomeIntegerAttr"),
    node->i("attr::SomeOtherIntegerAttr"), node->is("attr::AnIntArrayAttr"),
    node->f("attr::AFloatAttr"));
      }
    },
  */
  // Essentially this is just a map from the string IR symbol to a function to
  // be called that implements it. Those functions are also autogenerated by the
  // same macros in compiler.hpp and compiler.cpp.
  functionToImplementation = {
// Torch JIT api defines the attribute accessor as the following function names.
#define INT_VEC is
#define FLOAT_VEC fs
#define FLOAT f
#define INT i
#define BOOL i

// Useful NOP macro
#define NONE

// The arguments are processed by extracting the given type using the above
// accessors, the name is converted into "attr::NAME" which is what pytorch JIT
// expects for attribute accessing.
#define ARG(Type, Name)                                                        \
  , node->Type(c10::Symbol::fromQualString("attr::" #Name))
#define BODY_ARG(Name) NONE
// Create a function decl with the given call and arguments.
#define OP_DECL(name, function, unused, Args, unused2)                         \
  {name,                                                                       \
   [&](const std::vector<poptorch::TensorId> &inputs,                          \
       torch::jit::Node *node) { return compiler.function(inputs Args); }},

#include "popart_compiler/SupportedOperations.inc.h"

#undef BODY_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef INT_VEC
#undef FLOAT_VEC
#undef FLOAT
#undef INT
#undef BOOL
  }; // End map initalizer.
}

} // namespace

std::shared_ptr<poptorch::PoplarExecutable>
lowerToPopart(torch::jit::Graph &graph, std::vector<at::Tensor> &inTensors,
              std::vector<at::Tensor> &parameters, std::uint64_t steps,
              bool training, std::uint64_t replicationFactor,
              std::uint64_t gradientAccumulation, bool profile) {
  std::srand(std::time(nullptr));

  LowerToPopart lower_impl{
      graph,    inTensors,         parameters,           steps,
      training, replicationFactor, gradientAccumulation, profile};
  lower_impl.Lower();

  return lower_impl.Compile();
}

} // namespace poptorch
