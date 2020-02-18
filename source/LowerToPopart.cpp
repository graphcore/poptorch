#include "poptorch/LowerToPopart.hpp"
#include <iostream>
#include <torch/csrc/jit/ir.h>

#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/op/matmul.hpp>
#include <popart/tensors.hpp>

#include <iostream>

namespace poptorch {

namespace {

/*
 * Implementation of the lowering operation.
 */
class LowerToPopart {
public:
  LowerToPopart(torch::jit::Graph &g, InputTensorType &ins)
      : graph(g), inTensors(ins), opBuilder(popart::Builder::create()) {}

  void Lower();

private:
  torch::jit::Graph &graph;

  InputTensorType &inTensors;

  std::unique_ptr<popart::Builder> opBuilder;

  // Mapping between the SSA values of torch jit with the ssa values of popart.
  std::unordered_map<torch::jit::Value *, popart::TensorId> valueMap;

  void LowerParameters();

  void LowerBody();
};

/*
 * Static helper functions.
 */

std::string typeToPopart(const std::string &type) {
  if (type == "torch.float32") {
    return "FLOAT";
  }

  std::cerr << "UNIMPLEMENTED TYPE " << type << std::endl;
  return "UNIMPLEMENTED";
}

void LowerToPopart::Lower() {
  // Lower the tensor parameters of the graph to OpInputs.
  LowerParameters();

  // Lower the body of the graph.
  LowerBody();

  std::cout << "   PRINTING ONNX MODEL" << std::endl;
  std::cout << opBuilder->getModelProto() << std::endl;
  std::cout << "   DONE PRINTING ONNX MODEL" << std::endl;
}

// Lower the main body of the graph.
void LowerToPopart::LowerBody() {
    auto aiOnnx = opBuilder->aiOnnxOpset9();

  for (torch::jit::Node *node : graph.nodes()) {

    // Switch/lookup based on the actual int value.
    const std::string &bodyAsStr = node->kind().toDisplayString();

    // Get the torch jit SSA for the input/output values.
    torch::jit::Value* output = node->output();
    std::vector<popart::TensorId> inputs;
    std::transform(node->inputs().begin(), node->inputs().end(), std::back_inserter(inputs), [&](torch::jit::Value* val) {
        return valueMap[val];
    });


    if (bodyAsStr == "aten::t") {
        valueMap[output] = aiOnnx.transpose(inputs);
    } else if (bodyAsStr == "aten::matmul") {
        valueMap[output] = aiOnnx.matmul(inputs, "MatMul");

    } else if (bodyAsStr == "aten::add") {
        valueMap[output] = aiOnnx.add({inputs[0], inputs[1]}, "Add");
    } else if (bodyAsStr == "aten::relu") {
        valueMap[output] = aiOnnx.relu(inputs, "Relu");
    } else if (bodyAsStr == "prim::Constant") {
        // Ignore these constants I think we should eliminate them in earlier passes.
        valueMap[output] = "";
    }
  }
}

void LowerToPopart::LowerParameters() {
  // Lower user provided inputs first.
  std::size_t index = 0;
  for (auto pair : inTensors) {
    const std::string &type = pair.first;
    const std::vector<int64_t> &dims = pair.second;

    // Create the tensor info for our new tensor.
    popart::TensorInfo info{typeToPopart(type), dims};

    valueMap[graph.param_node()->outputs()[index]] =
        opBuilder->addInputTensor(info);

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

    // Convert the dimensions to popart supported type.
    std::vector<std::int64_t> popartDims(*dims.size());
    for (std::size_t i = 0; i < *dims.size(); ++i) {
      popartDims[i] = static_cast<std::int64_t>(*dims[i]);
    }

    // Unpack the elem type into its Popart type.
    std::string popartType = "FLOAT";
    // When we actually support more types unpack them here

    // Create the tensor info for our new tensor.
    popart::TensorInfo info{popartType, popartDims};

    valueMap[value] = opBuilder->addInputTensor(info);
  }

  //}
}

} // namespace

void lowerToPopart(torch::jit::Graph &graph, InputTensorType &inTensors) {
  LowerToPopart lower_impl{graph, inTensors};
  lower_impl.Lower();
}

} // namespace poptorch