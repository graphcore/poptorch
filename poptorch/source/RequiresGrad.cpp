// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/graph_node_list.h>
#include <torch/csrc/jit/ir/ir.h>

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/RequiresGrad.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

void fixRequiresGradFromDispatch(torch::jit::Graph *graph) {
  // For each output of each node in the graph.
  for (auto *node : graph->nodes()) {
    for (auto *output : node->outputs()) {
      auto tensor_type = output->type()->cast<c10::TensorType>();
      if (!tensor_type) {
        continue;
      }
      auto device = tensor_type->device();
      if (!device) {
        continue;
      }
      if (device->type() != at::DeviceType::IPU) {
        continue;
      }
      // If the output is an IPU floating-point tensor, check if any
      // of the inputs has requires_grad set, and update the Value if
      // needed.
      bool requires_grad = false;
      if (tensor_type->scalarType().has_value() &&
          c10::isFloatingType(tensor_type->scalarType().value())) {
        for (auto *input : node->inputs()) {
          if (input->requires_grad()) {
            requires_grad = true;
            break;
          }
        }
      }
      if (requires_grad != output->requires_grad()) {
        logging::trace("[requires_grad] Set requires_grad={} on node {}",
                       requires_grad, nodeToString(node));
        output->setType(tensor_type->withRequiresGrad(requires_grad));
      }
    }
  }
}

} // namespace poptorch
