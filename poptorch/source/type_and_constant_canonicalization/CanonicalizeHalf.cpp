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

  // Convert the inputs to the graph (i.e. both user input and parameters).
  void convertGraphInputs(const std::vector<at::Tensor> &in_tensors,
                          const std::vector<at::Tensor> &parameters);

  // Resolve types which are ambiguiously between half or float.
  void resolveHalfOrFloat();

private:
  static void convertTensorIfNeeded(const at::Tensor &tensor,
                                    torch::jit::Value *value);

  static bool atLeastOneUseHalf(const std::vector<torch::jit::Use> &uses);

  static void resolveValueType(torch::jit::Value *output,
                               at::ScalarType scalar_type);

  static void resolveNodeDtype(torch::jit::Node *node,
                               at::ScalarType scalar_type);

  // Duplicate poptorch::tensor_constants that are re-used throughout the graph
  // and have uses resolved to different types as a result of
  // resolveHalfOrFloat(). Constant re-use only happens in sufficiently large
  // graphs.
  void duplicateConstantsWithMixedHalfFloatUses();

  static std::vector<torch::jit::Node *>
  usersOfScalarType(at::ScalarType type,
                    const std::vector<torch::jit::Use> &uses);

  torch::jit::Graph *_graph;
};

void ConvertHalfImpl::convertTensorIfNeeded(const at::Tensor &tensor,
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
}

void ConvertHalfImpl::convertGraphInputs(
    const std::vector<at::Tensor> &in_tensors,
    const std::vector<at::Tensor> &parameters) {
  logging::LogContext ctx_func("convertGraphInputs");
  auto collapsed_inputs = collapsedGraphInputHierachy(_graph);

  std::size_t num_inputs = collapsed_inputs.size() - parameters.size();
  ERROR_ON(num_inputs != in_tensors.size());

  for (unsigned int idx = 0; idx < num_inputs; idx++) {
    torch::jit::Value *value = collapsed_inputs[idx];

    if (value == nullptr) {
      continue;
    }

    if (idx < num_inputs) {
      logging::LogContext ctx("processing " + nodeToString(value->node()));
      convertTensorIfNeeded(in_tensors[idx], value);
    } else {
      // Can't have tuples for parameters:
      ERROR_ON(value->type()->kind() != c10::TypeKind::TensorType);

      // Lower the other params (i.e the weights)
      const at::Tensor &tensor_as_param = parameters.at(idx - num_inputs);
      convertTensorIfNeeded(tensor_as_param, value);
    }
  }
}

bool ConvertHalfImpl::atLeastOneUseHalf(
    const std::vector<torch::jit::Use> &uses) {
  for (const auto &use : uses) {
    for (auto *output : use.user->outputs()) {
      auto type = output->type()->cast<c10::TensorType>();
      if (!type || !type->scalarType()) {
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
  for (auto *node : _graph->nodes().reverse()) {
    for (auto *output : node->outputs()) {
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
        node->s_(c10::Symbol::attr("to"), scalarTypeToOnnxString(new_type));
      }

      // Tensor constants may need retyping
      if (node->kind() == symbols::poptorch::tensor_constant ||
          node->kind() == symbols::poptorch::host_side_tensor_constant) {
        auto new_tensor = node->t(c10::attr::value).to(new_type).contiguous();
        node->t_(c10::attr::value, new_tensor);
      }

      if (node->hasAttribute(c10::attr::dtype)) {
        resolveNodeDtype(node, new_type);
      }
    }
  }
  duplicateConstantsWithMixedHalfFloatUses();
}

std::vector<torch::jit::Node *>
ConvertHalfImpl::usersOfScalarType(const at::ScalarType type,
                                   const std::vector<torch::jit::Use> &uses) {
  std::vector<torch::jit::Node *> users;
  for (const auto &use : uses) {
    for (auto *output : use.user->outputs()) {
      auto output_type = output->type()->cast<c10::TensorType>();
      if (!output_type || !output_type->scalarType()) {
        continue;
      }
      if ((*output_type->scalarType()) == type) {
        users.push_back(use.user);
      }
    }
  }
  return users;
}

void ConvertHalfImpl::duplicateConstantsWithMixedHalfFloatUses() {
  logging::LogContext ctx_func("DuplicateConstantsWithMixedHalfFloatUses");

  for (auto *node : _graph->nodes()) {
    if (node->kind() != symbols::poptorch::tensor_constant) {
      continue;
    }

    // Take only half/float constants into account as we are resolving the
    // output of a half/float resolution pass.
    auto tensor_type = node->output()->type()->cast<c10::TensorType>();
    if (!tensor_type || !tensor_type->scalarType()) {
      continue;
    }
    auto scalar_type = *(tensor_type->scalarType());
    if (scalar_type != at::ScalarType::Half &&
        scalar_type != at::ScalarType::Float) {
      continue;
    }

    auto invalid_type = scalar_type == at::ScalarType::Float
                            ? at::ScalarType::Half
                            : at::ScalarType::Float;

    auto invalid_nodes =
        usersOfScalarType(invalid_type, node->output()->uses());
    if (invalid_nodes.empty()) {
      continue;
    }

    // Duplicate the constant.
    auto *new_node = _graph->createClone(
        node, [](torch::jit::Value *value) { return value; });
    logging::trace("Duplicating constant {} with {}", nodeToString(node),
                   nodeToString(new_node));

    new_node->output()->setType(
        new_node->output()->type()->expect<c10::TensorType>()->withScalarType(
            invalid_type));
    auto new_tensor =
        new_node->t(c10::attr::value).to(invalid_type).contiguous();
    new_node->t_(c10::attr::value, new_tensor);

    new_node->insertBefore(node);
    for (auto *invalid_node : invalid_nodes) {
      invalid_node->replaceInputWith(node->output(), new_node->output());
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
