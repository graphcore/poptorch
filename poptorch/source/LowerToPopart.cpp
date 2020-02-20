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
  LowerToPopart(torch::jit::Graph &g, std::vector<at::Tensor> &ins)
      : graph(g), inTensors(ins) {}

  void Lower();

  at::IValue CompileAndRun();

private:
  torch::jit::Graph &graph;

  std::vector<at::Tensor> &inTensors;

  // Mapping between the SSA values of torch jit with the ssa values of popart.
  std::unordered_map<torch::jit::Value *, poptorch::TensorId> valueMap;
  std::unordered_map<poptorch::TensorId, at::Tensor *> inputMap;

  std::list<poptorch::TensorId> outputs;


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
  }

  std::cerr << "UNIMPLEMENTED TYPE " << type << std::endl;
  return "UNIMPLEMENTED";
}

/*
 * Lower to popart impl.
 */

at::IValue LowerToPopart::CompileAndRun() {
  for (auto &pair : inputMap) {
    // Convert to correct data type.
    std::vector<std::int64_t> popartDims(pair.second->sizes().size());

    std::transform(pair.second->sizes().begin(), pair.second->sizes().end(),
                   popartDims.begin(), [](std::int64_t i) { return i; });

    compiler.SetUpInputOp(
        pair.first, static_cast<float *>(pair.second->data_ptr()), popartDims);
  }

  // Init the session after we compile the graph but before we create the output
  // buffers so we can get the size.
  compiler.InitSession();

  // Temp buffers for the output state.
  std::map<poptorch::TensorId, at::IValue> torchOutputs;

  // Set up the outputs.
  for (poptorch::TensorId id : outputs) {
    std::vector<std::int64_t> dims = compiler.GetSize(id);

    // Create the torch tensor and use its memory for the popart tensor.
    torchOutputs[id] = at::empty({dims});
    float *dataPtr = (float *)torchOutputs[id].toTensor().data_ptr();

    compiler.SetUpOutputOp(id, dataPtr, dims);
  }

  compiler.Run();

  // Return the outputs as pytorch tensors to the user.
  for (auto &pair : torchOutputs) {

    std::cout << pair.second.toTensor().data_ptr() << "  "
              << *(float *)pair.second.toTensor().data_ptr() << std::endl;

    return pair.second;
  }
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
    outputs.push_back(valueMap[value]);
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

    valueMap[output] = compiler.BuildOp(bodyAsStr.c_str(), inputs);
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
    inputMap[id] = &tensor;

    ++index;
  }

  // Then lower the other params (I.E the weights.)
  for (torch::jit::Value *value : graph.param_node()->outputs()) {
    // Skip the values already added (I.E)
    if (valueMap.find(value) != valueMap.end())
      continue;

    // Get the underlaying type, only tensor for now.
    c10::TensorTypePtr asTensor = value->type()->cast<c10::TensorType>();
    c10::VaryingShape dims = asTensor->sizes();
    at::ScalarType elem_type = *asTensor->scalarType();

    // Random number generator with a normal distribution mean of 0 and variance
    // of 1.
    std::random_device random{};
    std::normal_distribution<float> distribution{0, 1};

    // Set the inital value to be random numbers.
    std::vector<float> initalValue(*asTensor->numel());

    std::cout << "Weight random init distribution: ";
    std::transform(initalValue.begin(), initalValue.end(), initalValue.begin(),
                   [&](float) {
                     // Kaiming random initalization.
                     float x = distribution(random) * std::sqrt(2.0f / (10.0f));
                     std::cout << x << " ";
                     return x;
                   });
    std::cout << std::endl;

    // Convert the dimensions to popart supported type.
    std::vector<std::int64_t> popartDims(*dims.size());
    for (std::size_t i = 0; i < *dims.size(); ++i) {
      popartDims[i] = static_cast<std::int64_t>(*dims[i]);
    }

    // Unpack the elem type into its Popart type.
    std::string popartType = "FLOAT";
    // When we actually support more types unpack them here

    valueMap[value] = compiler.AddInitializedInputTensor(
        "Weight", popartType.c_str(), popartDims,
        static_cast<void *>(initalValue.data()));
  }
}

} // namespace

at::IValue lowerToPopart(torch::jit::Graph &graph,
                         std::vector<at::Tensor> &inTensors) {
  std::srand(std::time(nullptr));

  LowerToPopart lower_impl{graph, inTensors};
  lower_impl.Lower();

  return lower_impl.CompileAndRun();
}

} // namespace poptorch
