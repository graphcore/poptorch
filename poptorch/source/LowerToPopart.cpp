#include "poptorch/LowerToPopart.hpp"
#include <iostream>

#include <torch/csrc/jit/ir.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <list>
#include <random>

#include <popart_compiler/Compiler.hpp>

namespace poptorch {

namespace {

/*
 * Implementation of the lowering operation.
 */
class LowerToPopart {
public:
  LowerToPopart(torch::jit::Graph &g, std::vector<at::Tensor> &ins,
                std::vector<at::Tensor> &params, std::uint64_t steps,
                bool training);

  void Lower();

  std::shared_ptr<poptorch::PoplarExecutable> Compile();

private:
  torch::jit::Graph &graph;

  std::vector<at::Tensor> &inTensors;

  std::vector<at::Tensor> &parameters;

  std::vector<poptorch::TensorId> inputTensorHooks;

  std::vector<poptorch::TensorId> outputTensorHooks;

  // Mapping between the SSA values of torch jit with the ssa values of popart.
  std::unordered_map<torch::jit::Value *, poptorch::TensorId> valueMap;

  using FunctionType = std::function<poptorch::TensorId(
      const std::vector<poptorch::TensorId> &inputs, torch::jit::Node *)>;
  std::unordered_map<std::string, FunctionType> functionToImplementation;

  poptorch::Compiler compiler;

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
  } else if (type == at::ScalarType::Long) {
    return "INT64";
  } else if (type == at::ScalarType::Int) {
    return "INT32";
  }

  std::cerr << "UNIMPLEMENTED TYPE " << type << std::endl;
  return "UNIMPLEMENTED";
}

/*
 * Lower to popart impl.
 */
std::shared_ptr<poptorch::PoplarExecutable> LowerToPopart::Compile() {
  // Init the session, this also involves compiling to poplar.
  compiler.InitSession();

  return std::make_shared<poptorch::PoplarExecutable>(
      std::move(compiler), std::move(inputTensorHooks),
      std::move(outputTensorHooks));
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
    compiler.AddOutput(valueMap[value]);
    outputTensorHooks.push_back(valueMap[value]);
  }
}

// Lower the main body of the graph.
void LowerToPopart::LowerBody() {
  for (torch::jit::Node *node : graph.nodes()) {
    // Switch/lookup based on the actual int value.
    const std::string &bodyAsStr = node->kind().toDisplayString();

    // Get the torch jit SSA for the input/output values.
    torch::jit::Value *output = node->output();
    std::vector<poptorch::TensorId> inputs;
    std::transform(node->inputs().begin(), node->inputs().end(),
                   std::back_inserter(inputs),
                   [&](torch::jit::Value *val) { return valueMap[val]; });

    auto itr = functionToImplementation.find(bodyAsStr);

    if (itr != functionToImplementation.end()) {
      // Call the callback.
      valueMap[output] = itr->second(inputs, node);
    } else {
      std::cerr << "ERROR: couldn't find a registered operation for node "
                << std::endl;
      node->dump();
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
    valueMap[graph.param_node()->outputs()[index]] = id;
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

    valueMap[value] = compiler.AddInitializedInputTensor(
        "Weight", popartType.c_str(), dims, tensorAsParam.data_ptr());

    paramIndex++;
  }
}

LowerToPopart::LowerToPopart(torch::jit::Graph &g, std::vector<at::Tensor> &ins,
                             std::vector<at::Tensor> &params,
                             std::uint64_t steps, bool training)
    : graph(g), inTensors(ins), parameters(params), compiler({training, steps}) {
  // Init the function implementation map.

  functionToImplementation = {
#define INT_VEC is
#define FLOAT f
#define INT i
#define BOOL i
#define NONE
#define ARG(Type, Name)                                                        \
  , node->Type(c10::Symbol::fromQualString("attr::" #Name))
#define BODY_ARG(Name) NONE
// Create a function decl with the given call and arguments.
#define OP_DECL(name, function, unused, Args, unused2, VariadicIndex)          \
  {name,                                                                       \
   [&](const std::vector<poptorch::TensorId> &inputs,                          \
       torch::jit::Node *node) { return compiler.function(inputs Args); }},

#include "popart_compiler/SupportedOperations.inc.h"

#undef BODY_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef INT_VEC
#undef FLOAT
#undef INT
#undef BOOL
  }; // End map initalizer.
}

} // namespace

std::shared_ptr<poptorch::PoplarExecutable>
lowerToPopart(torch::jit::Graph &graph, std::vector<at::Tensor> &inTensors,
              std::vector<at::Tensor> &parameters, std::uint64_t steps,
              bool training) {
  std::srand(std::time(nullptr));

  LowerToPopart lower_impl{graph, inTensors, parameters, steps, training};
  lower_impl.Lower();

  return lower_impl.Compile();
}

} // namespace poptorch
