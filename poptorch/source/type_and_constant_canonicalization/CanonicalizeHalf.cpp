// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "popart_compiler/PopartEnums.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {

namespace {
class ConvertHalfImpl {
public:
  explicit ConvertHalfImpl(torch::jit::Graph *g) : _graph(g) {}

  // Convert the inputs to the graph (I.E both user input and parameters.)
  void convertGraphInputs(const std::vector<at::Tensor> &in_tensors,
                          const std::vector<at::Tensor> &parameters);

  // Resolve types which are ambiguiously between half or float
  void resolveHalfOrFloat();

private:
  static c10::TypePtr
  convertValueIfNeeded(torch::jit::Value *value,
                       const std::vector<at::Tensor> &in_tensors,
                       std::vector<at::Tensor>::const_iterator *input_iterator,
                       std::int64_t input_index);

  static c10::TypePtr convertTensorIfNeeded(const at::Tensor &tensor,
                                            torch::jit::Value *value);

  static bool atLeastOneUseHalf(const std::vector<torch::jit::Use> &uses);

  static void resolveValueType(torch::jit::Value *output,
                               at::ScalarType scalar_type);

  static void resolveNodeDtype(torch::jit::Node *node,
                               at::ScalarType scalar_type);

  torch::jit::Graph *_graph;
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
  }
  return new_type;
}

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

bool ConvertHalfImpl::atLeastOneUseHalf(
    const std::vector<torch::jit::Use> &uses) {
  for (const auto &use : uses) {
    for (auto output : use.user->outputs()) {
      auto type = output->type()->cast<c10::TensorType>();
      if (!type) {
        continue;
      }
      if ((*type->scalarType()) == at::ScalarType::Half) {
        return true;
      }
    }
  }
  return false;
}

void ConvertHalfImpl::resolveValueType(torch::jit::Value *output,
                                       const at::ScalarType scalar_type) {
  auto output_type = output->type()->expect<c10::TensorType>();
  ERROR_ON(*output_type->scalarType() != HALF_OR_FLOAT);
  output->setType(output_type->withScalarType(scalar_type));
}

void ConvertHalfImpl::resolveNodeDtype(torch::jit::Node *node,
                                       const at::ScalarType scalar_type) {
  if (node->kindOf(c10::attr::dtype) == torch::jit::AttributeKind::i) {
    node->i_(c10::attr::dtype,
             dtypeIntFromOnnxStr(scalarTypeToOnnxString(scalar_type).c_str()));
  } else {
    node->s_(c10::attr::dtype, scalarTypeToOnnxString(scalar_type));
  }
}

void ConvertHalfImpl::resolveHalfOrFloat() {
  // Iterate the graph in reverse
  for (auto node : _graph->nodes().reverse()) {
    for (auto output : node->outputs()) {
      auto output_type = output->type()->cast<c10::TensorType>();
      if (!output_type) {
        continue;
      }

      auto scalar_type = output_type->scalarType();

      if (!scalar_type) {
        continue;
      }
      if (*scalar_type != HALF_OR_FLOAT) {
        continue;
      }

      // Resolve to half if at least one use is half
      auto new_type = atLeastOneUseHalf(output->uses()) ? at::ScalarType::Half
                                                        : at::ScalarType::Float;
      resolveValueType(output, new_type);

      // Some nodes need an attribute changing to match
      if (node->kind() == symbols::popart::cast) {
        node->s_(c10::Symbol::fromQualString("attr::to"),
                 scalarTypeToOnnxString(new_type));
      }

      // Tensor constants may need retyping
      if (node->kind() == symbols::poptorch::tensor_constant) {
        auto new_tensor = node->t(c10::attr::value).to(new_type);
        node->t_(c10::attr::value, new_tensor);
      }

      if (node->hasAttribute(c10::attr::dtype)) {
        resolveNodeDtype(node, new_type);
      }
    }
  }
}
} // namespace

void canonicaliseHalfInputs(torch::jit::Graph *graph,
                            const std::vector<at::Tensor> &in_tensors,
                            const std::vector<at::Tensor> &parameters) {
  ConvertHalfImpl impl{graph};
  impl.convertGraphInputs(in_tensors, parameters);
}

void resolveHalfOrFloat(torch::jit::Graph *graph) {
  ConvertHalfImpl impl{graph};
  impl.resolveHalfOrFloat();
}

} // namespace poptorch
