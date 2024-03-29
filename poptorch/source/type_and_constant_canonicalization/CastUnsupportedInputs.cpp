// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "torch/csrc/jit/ir/ir.h"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {
namespace {
void processInputTensor(torch::jit::Graph *graph, torch::jit::Value *input) {
  auto tensor_type = input->type()->expect<c10::TensorType>();
  auto current_type = tensor_type->scalarType().value();

  at::ScalarType new_type = coerceToSupportedType(current_type);

  if (current_type == at::ScalarType::BFloat16) {
    new_type = at::ScalarType::Half;
  } else if (new_type == current_type) {
    // No need for a host side cast
    return;
  }

  auto *earliest_user = findEarliestUser(input);
  if (earliest_user == nullptr) {
    logging::warn("Graph contains an unused input %{} : {}", input->debugName(),
                  *tensor_type);
    return;
  }

  // This is an identity op but used just to make sure the implicit cast
  // does not end up promoting to a Double/Long
  auto *new_node = graph->create(symbols::poptorch::host_side_cast);

  insertNodeBeforeNode(new_node, earliest_user);
  input->replaceAllUsesWith(new_node->output());
  new_node->addInput(input);

  new_node->output()->setType(tensor_type->withScalarType(new_type));
}
} // namespace

void castUnsupportedInputs(torch::jit::Graph *graph) {
  auto collapsed_inputs = collapsedGraphInputHierachy(graph);

  for (auto *input : collapsed_inputs) {
    if (input != nullptr) {
      processInputTensor(graph, input);
    }
  }
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
