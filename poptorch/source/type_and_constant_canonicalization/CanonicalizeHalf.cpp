// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"

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
  c10::TypePtr convertTensorIfNeeded(const at::Tensor &tensor,
                                     torch::jit::Value *value);
  c10::TypePtr
  convertValueIfNeeded(torch::jit::Value *value,
                       const std::vector<at::Tensor> &in_tensors,
                       std::vector<at::Tensor>::const_iterator *input_iterator,
                       std::int64_t input_index);

  torch::jit::Graph *_graph;

  std::unordered_set<torch::jit::Value *> _converted_tensors;
};

c10::TypePtr ConvertHalfImpl::convertValueIfNeeded(
    torch::jit::Value *value, const std::vector<at::Tensor> &in_tensors,
    std::vector<at::Tensor>::const_iterator *input_iterator,
    std::int64_t input_index) {
  c10::TypePtr new_type = nullptr;
  switch (value->type()->kind()) {
  case c10::TypeKind::TensorType: {
    ERROR_ON(*input_iterator == in_tensors.end());
    new_type = convertTensorIfNeeded(**input_iterator, value);
    (*input_iterator)++;
    break;
  }
  case c10::TypeKind::TupleType: {
    auto tuple = value->type()->expect<c10::TupleType>();
    std::vector<c10::TypePtr> new_types;
    bool type_changed = false;

    // Unused tuple
    if (!value->hasUses()) {
      break;
    }

    // Until we encounter something else we only support TupleUnpack
    ERROR_ON(value->uses().size() != 1);
    auto user = value->uses()[0].user;
    ERROR_ON(user->kind() != c10::prim::TupleUnpack);
    ERROR_ON(tuple->elements().size() != user->outputs().size());
    for (auto output : user->outputs()) {
      auto changed_type =
          convertValueIfNeeded(output, in_tensors, input_iterator, input_index);
      if (changed_type) {
        new_types.push_back(changed_type);
        type_changed = true;
      } else {
        new_types.push_back(output->type());
      }
    }
    if (type_changed) {
      new_type = c10::TupleType::create(new_types);
      value->setType(new_type);
    }
    break;
  }
  default:
    ERROR("Unsupported parameter type '"
          << c10::typeKindToString(value->type()->kind()) << "' for input "
          << input_index);
  }
  return new_type;
}

c10::TypePtr ConvertHalfImpl::convertTensorIfNeeded(const at::Tensor &tensor,
                                                    torch::jit::Value *value) {
  c10::TypePtr new_type = nullptr;
  // If the actual input tensor is half.
  if (tensor.scalar_type() == at::ScalarType::Half) {
    logging::trace("Converting parameter {} to half",
                   nodeToString(value->node()));
    c10::TensorTypePtr as_tensor = value->type()->expect<c10::TensorType>();
    new_type = as_tensor->withScalarType(c10::ScalarType::Half);
    value->setType(new_type);
    // Add it to the list of converted tensors.
    _converted_tensors.insert(value);
  }
  return new_type;
}

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
  }

  c10::TypePtr type = tensor->type();

  c10::TensorTypePtr as_tensor = type->cast<c10::TensorType>();

  // The general case of converting an IR tensor.
  if (!as_tensor || !as_tensor->scalarType() ||
      *as_tensor->scalarType() != at::ScalarType::Float) {
    return false;
  }

  tensor->setType(as_tensor->withScalarType(c10::ScalarType::Half));

  if (node->kind() == symbols::poptorch::tensor_constant) {
    node->t_(c10::attr::value,
             node->t(c10::attr::value).to(at::ScalarType::Half));
  }

  return true;
}

/*
 * Impl
 */
void ConvertHalfImpl::convertGraphInputs(
    const std::vector<at::Tensor> &in_tensors,
    const std::vector<at::Tensor> &parameters) {
  std::size_t index = 0;
  std::size_t num_inputs =
      _graph->param_node()->outputs().size() - parameters.size();
  auto tensor_it = in_tensors.begin();

  // For each input in the IR view.
  for (torch::jit::Value *value : _graph->inputs()) {
    if (index < num_inputs) {
      // Lower user provided input
      ERROR_ON(value->node()->kind() != c10::prim::Param);
      convertValueIfNeeded(value, in_tensors, &tensor_it, index);
    } else {
      ERROR_ON_MSG(tensor_it != in_tensors.end(),
                   "Not all the input tensors have been used");
      // Can't have tuples for parameters:
      ERROR_ON(value->type()->kind() != c10::TypeKind::TensorType);
      // Lower the other params (i.e the weights)
      const at::Tensor &tensor_as_param = parameters.at(index - num_inputs);
      convertTensorIfNeeded(tensor_as_param, value);
    }
    ++index;
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
