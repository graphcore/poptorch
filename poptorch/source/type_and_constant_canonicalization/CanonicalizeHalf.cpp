// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/PopartCanonicalization.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {

namespace {
/*
 * 1. Convert any input to half if the provided tensor is half.
 * 2. Look at uses of that value and change the return type of that tensor to
 * half if it was previously float.
 * 3. Check if any of other the incoming operands are cast or constant and
 * change them to half if they were float.
 * 4. Repeat from 2. with the updated ops until there is nothing left to
 * convert.
 */
class ConvertHalfImpl {
public:
  explicit ConvertHalfImpl(torch::jit::Graph *g) : _graph(g) {}

  // Convert the inputs to the graph (I.E both user input and parameters.)
  void convertGraphInputs(const std::vector<at::Tensor> &in_tensors,
                          const std::vector<at::Tensor> &parameters);

  // Convert the uses of those inputs recursively.
  void convertUses();

private:
  torch::jit::Graph *_graph;

  std::unordered_set<torch::jit::Value *> _converted_tensors;
};

/*
 * Static helper functions.
 */

bool maybeConvertTensor(torch::jit::Value *tensor) {
  // Check if a node can be converted directly (casts and constants).
  torch::jit::Node *node = tensor->node();

  const std::string float_string{"FLOAT"};
  const std::string half_string{"FLOAT16"};

  if (node->kind() == symbols::popart::cast) {
    if (float_string == node->s(c10::Symbol::fromQualString("attr::to"))) {
      node->s_(c10::Symbol::fromQualString("attr::to"), "FLOAT16");
      return true;
    }
    if (half_string == node->s(c10::Symbol::fromQualString("attr::to"))) {
      // Don't propagate through half casts, implies node above is not half by
      // design.
      return false;
    }
  } else if (node->kind() == symbols::poptorch::float_constant) {
    // Tell the backend this constant should be generated as half.
    node->i_(c10::Symbol::fromQualString("attr::isHalf"), 1);
    return true;
  }

  c10::TypePtr type = tensor->type();

  c10::TensorTypePtr as_tensor = type->cast<c10::TensorType>();

  // The general case of converting an IR tensor.
  if (!as_tensor || !as_tensor->scalarType() ||
      *as_tensor->scalarType() != at::ScalarType::Float) {
    return false;
  }

  tensor->setType(as_tensor->withScalarType(c10::ScalarType::Half));

  return true;
}

/*
 * Impl
 */
void ConvertHalfImpl::convertGraphInputs(
    const std::vector<at::Tensor> &in_tensors,
    const std::vector<at::Tensor> &parameters) {
  // For each input in the IR view.
  for (std::size_t index = 0; index < _graph->inputs().size(); ++index) {
    // Take the tensor from either the parameters or the input tensor. They are
    // flattened in the IR at this point.
    const at::Tensor &tensor = index >= in_tensors.size()
                                   ? parameters[index - in_tensors.size()]
                                   : in_tensors[index];

    // If the actual input tensor is half.
    if (tensor.scalar_type() == at::ScalarType::Half) {
      logging::trace("Converting parameter {} to half", index);
      // Convert the IR type to half.
      c10::TypePtr type = _graph->inputs()[index]->type();
      c10::TensorTypePtr as_tensor = type->expect<c10::TensorType>();
      _graph->inputs()[index]->setType(
          as_tensor->withScalarType(c10::ScalarType::Half));

      // Add it to the list of converted tensors.
      _converted_tensors.insert(_graph->inputs()[index]);
    }
  }
}

void ConvertHalfImpl::convertUses() {
  while (!_converted_tensors.empty()) {
    // Pop from the work list.
    auto itr = _converted_tensors.begin();
    torch::jit::Value *tensor = *itr;
    _converted_tensors.erase(itr);

    // Check the users and convert if need be.
    for (torch::jit::Use use : tensor->uses()) {
      torch::jit::Node *node = use.user;

      for (torch::jit::Value *output : node->outputs()) {
        if (maybeConvertTensor(output)) {
          _converted_tensors.insert(output);
        }
      }
    }

    // Check the inputs and list them all for conversion.
    for (torch::jit::Value *input : tensor->node()->inputs()) {
      if (maybeConvertTensor(input)) {
        _converted_tensors.insert(input);
      }
    }
  }
}

} // namespace

void canonicaliseHalf(torch::jit::Graph *graph,
                      const std::vector<at::Tensor> &in_tensors,
                      const std::vector<at::Tensor> &parameters) {
  ConvertHalfImpl impl{graph};
  impl.convertGraphInputs(in_tensors, parameters);
  impl.convertUses();
}

} // namespace poptorch
