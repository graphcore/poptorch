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
  // Ignore nodes for which the type is inconsequential
  if (node->kind() == c10::aten::argmax || node->kind() == c10::aten::argmin ||
      node->kind() == c10::aten::contiguous ||
      node->kind() == c10::aten::chunk || node->kind() == c10::aten::detach ||
      node->kind() == c10::aten::expand ||
      node->kind() == c10::aten::expand_as ||
      node->kind() == c10::aten::flatten || node->kind() == c10::aten::ones ||
      node->kind() == c10::aten::ones || node->kind() == c10::aten::permute ||
      node->kind() == c10::aten::reshape || node->kind() == c10::aten::select ||
      node->kind() == c10::aten::slice || node->kind() == c10::aten::split ||
      node->kind() == c10::aten::stack || node->kind() == c10::aten::squeeze ||
      node->kind() == c10::aten::transpose ||
      node->kind() == c10::aten::unsqueeze ||
      node->kind() == c10::aten::upsample_nearest1d ||
      node->kind() == c10::aten::upsample_nearest2d ||
      node->kind() == c10::aten::upsample_nearest3d ||
      node->kind() == c10::aten::upsample_linear1d ||
      node->kind() == c10::aten::upsample_bilinear2d ||
      node->kind() == c10::aten::upsample_trilinear3d ||
      node->kind() == c10::aten::upsample_bicubic2d ||
      node->kind() == c10::aten::view || node->kind() == c10::aten::zeros ||
      node->kind() == c10::prim::NumToTensor) {
    return;
  }

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
    warnNonNativeSupport(node, torch_type_str);
    output->setType(current_type->withScalarType(replacement_dtype));
  }

  // Ensure no casting to it
  if (node->kind() == c10::aten::to) {
    // Possible locations of dtype int depending on the aten::to arity
    auto num_inputs = node->inputs().size();
    size_t dtype_index = 0;

    if (num_inputs == 5 || num_inputs == 8) {
      dtype_index = 1;
    } else if (num_inputs == 6) {
      dtype_index = 2;
    } else {
      // Must be another aten::to signature
      return;
    }

    auto int_type = node->input(dtype_index)->type()->cast<c10::IntType>();
    ERROR_ON_MSG(!int_type, "Expected integer type as dtype input at index "
                                << dtype_index << " for "
                                << nodeToString(node));

    auto replacement = static_cast<int>(replacement_dtype);
    auto input = node->input(dtype_index)->node();
    logging::warn("Replacing {} with {} for {}", input->i(c10::attr::value),
                  replacement, nodeToString(input));
    input->i_(c10::attr::value, replacement);
  }
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
