// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {

void makeConstantIntParams(torch::jit::Graph *graph,
                           std::vector<std::string> &parameter_names,
                           std::vector<at::Tensor> &traced_parameter_tensors) {
  logging::LogContext ctx_func("makeConstantIntParams");
  // _parameters in Lower to popart is traced_parameter_tensors here.
  std::size_t num_inputs =
      graph->inputs().size() - traced_parameter_tensors.size();

  std::vector<std::size_t> to_delete;
  std::size_t index = 0;
  for (torch::jit::Value *value : graph->inputs()) {
    if (index < num_inputs) {
      index++;
      continue;
    }

    logging::LogContext ctx("processing " +
                            parameter_names[index - num_inputs]);

    // _parameters in Lower to popart is traced_parameter_tensors here.
    auto tensor = traced_parameter_tensors[index - num_inputs];

    if (value->type()->kind() == c10::TypeKind::TensorType) {
      auto tensor_type = value->type()->expect<c10::TensorType>();
      auto current_type = tensor_type->scalarType().value();

      if (!c10::isFloatingType(current_type)) {
        // Some nodes might not be used, we skip them if so.
        torch::jit::Node *earliest_user = findEarliestUser(value);
        if (earliest_user == nullptr) {
          continue;
        }

        torch::jit::WithInsertPoint insert_point(earliest_user);

        if (current_type == at::ScalarType::Long) {
          tensor = tensor.to(at::ScalarType::Int);
        }

        auto *new_node = tensorToConstant(graph, tensor);
        logging::trace("makeConstantIntParams removing graph input %{} and "
                       "adding constant value %{}",
                       value->debugName(), new_node->output()->debugName());

        for (size_t use_idx = 0; use_idx < value->uses().size(); use_idx++) {
          auto u = value->uses()[use_idx];
          u.user->replaceInput(u.offset, new_node->output());
          use_idx--;
        }

        ERROR_ON(!value->uses().empty());
        to_delete.push_back(index);
      }
    } else {
      // There is no known case of a parameter or buffer being a type other than
      // TensorType after tracing. Log a warning to assist debugging if a case
      // is found.
      logging::warn("Non tensor parameter/buffer identified: {}",
                    parameter_names[index - num_inputs]);
    }

    index++;
  }

  // Delete highest index first not to invalidate the later indices.
  ERROR_ON(!std::is_sorted(to_delete.begin(), to_delete.end()));

  for (auto it = to_delete.rbegin(); it != to_delete.rend(); ++it) {
    size_t del_i = *it;
    size_t del_i_params = del_i - num_inputs;

    parameter_names.erase(parameter_names.begin() + del_i_params);
    traced_parameter_tensors.erase(traced_parameter_tensors.begin() +
                                   del_i_params);
    graph->eraseInput(del_i);
  }
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
