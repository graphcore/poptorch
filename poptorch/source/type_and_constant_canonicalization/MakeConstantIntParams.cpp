// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {

void makeConstantIntParams(torch::jit::Graph *graph,
                           const std::vector<std::string> &parameter_names,
                           const std::vector<at::Tensor> &traced_tensors) {
  //_parameters in Lower to popart is traced_tensors here
  std::size_t num_inputs = graph->inputs().size() - traced_tensors.size();

  std::size_t index = 0;
  for (torch::jit::Value *value : graph->inputs()) {
    if (index < num_inputs) {
      index++;
      continue;
    }

    logging::LogContext ctx("makeConstantIntParams processing " +
                            parameter_names[index - num_inputs]);

    //_parameters in Lower to popart is traced_tensors here
    auto tensor = traced_tensors[index - num_inputs];

    switch (value->type()->kind()) {
    case c10::TypeKind::TensorType: {
      auto tensor_type = value->type()->expect<c10::TensorType>();
      auto current_type = tensor_type->scalarType().value();

      if (!c10::isFloatingType(current_type)) {
        torch::jit::WithInsertPoint insert_point(findEarliestUser(value));

        if (current_type == at::ScalarType::Long) {
          tensor = tensor.to(at::ScalarType::Int);
        }

        auto new_node = tensorToConstant(graph, tensor);
        for (size_t use_idx = 0; use_idx < value->uses().size(); use_idx++) {
          auto u = value->uses()[use_idx];
          u.user->replaceInput(u.offset, new_node->output());
          use_idx--;
        }

        ERROR_ON(!value->uses().empty());
      }

      break;
    }
    default:
      // Tuples etc coming soon
      break;
    }

    index++;
  }
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
