// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/jit/ir/ir.h>

#include <sstream>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/Utils.hpp"

#include "poptorch/TypeAndConstantCanonicalization.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {

namespace {
constexpr bool supportedType(const at::ScalarType type) {
  return (type == at::ScalarType::Int || type == at::ScalarType::Long ||
          type == at::ScalarType::Half || type == at::ScalarType::Float ||
          type == at::ScalarType::Double || type == at::ScalarType::Bool ||
          type == at::ScalarType::BFloat16);
}

void warnNonNativeSupport(torch::jit::Node *node,
                          const char *unsupported_type) {
  logging::warn("{}: {} is not supported natively on IPU, loss of "
                "range/precision may occur",
                nodeToString(node), unsupported_type);
}

void maybeReplaceOutputType(torch::jit::Node *node, torch::jit::Value *output,
                            c10::TensorType *current_type,
                            const at::ScalarType unsupported_dtype,
                            const at::ScalarType replacement_dtype,
                            const char *torch_type_str) {
  if (current_type->scalarType() != unsupported_dtype) {
    return;
  }

  // Constants will be retyped later
  if (node->kind() != c10::prim::Constant) {
    output->setType(current_type->withScalarType(replacement_dtype));
  }

  // Ensure no casting to it
  if (node->kind() == c10::aten::to) {
    // Possible locations of dtype int
    for (size_t idx = 1; idx < node->inputs().size(); idx++) {
      if (node->input(idx)->type()->cast<c10::IntType>()) {
        node->input(idx)->node()->i_(c10::attr::value,
                                     static_cast<int>(replacement_dtype));
      }
    }
  }

  warnNonNativeSupport(node, torch_type_str);
}

void checkAndChangeOutputTypesForOutput(torch::jit::Node *node,
                                        torch::jit::Value *output) {
  auto tensor_type = output->type()->cast<c10::TensorType>();

  // Ignore other return types e.g.  NumberTypes for constants
  if (!tensor_type) {
    return;
  }

  ERROR_ON_MSG(!tensor_type->scalarType().has_value(),
               "Returning an unknown tensor dtype is not supported.\n");

  ERROR_ON_MSG(!supportedType(*tensor_type->scalarType()),
               "Returning a torch."
                   << torch::getTHPDtype(*tensor_type->scalarType())->name
                   << " is not supported.\n");

  maybeReplaceOutputType(node, output, tensor_type.get(),
                         at::ScalarType::Double, at::ScalarType::Float,
                         "torch.float64");
  maybeReplaceOutputType(node, output, tensor_type.get(), at::ScalarType::Long,
                         at::ScalarType::Int, "torch.int64");
  maybeReplaceOutputType(node, output, tensor_type.get(),
                         at::ScalarType::BFloat16, at::ScalarType::Half,
                         "torch.bfloat16");
}
} // namespace

void checkAndChangeOutputTypes(torch::jit::Graph *graph) {
  for (auto n : graph->nodes()) {
    // Unpacks will happen before a host side cast, so ignore here
    if (n->kind() == c10::prim::TupleUnpack ||
        n->kind() == c10::prim::ListUnpack) {
      continue;
    }

    logging::LogContext ctx("CheckAndChangeOutputTypes processing " +
                            nodeToString(n));

    for (auto output : n->outputs()) {
      logging::LogContext ctx_2(output->debugName());

      checkAndChangeOutputTypesForOutput(n, output);
    }
  }
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
