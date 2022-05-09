// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/jit/ir/ir.h>

#include <sstream>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {

namespace {
constexpr bool supportedType(const at::ScalarType type) {
  return (type == at::ScalarType::Int || type == at::ScalarType::Long ||
          type == at::ScalarType::Half || type == at::ScalarType::Float ||
          type == at::ScalarType::Double || type == at::ScalarType::Bool ||
          type == at::ScalarType::BFloat16 || type == at::ScalarType::Char ||
          type == at::ScalarType::Byte || type == at::ScalarType::Short);
}

bool isBeforeHostSideCast(const torch::jit::Node *n) {
  if (n->kind() == c10::prim::TupleUnpack ||
      n->kind() == c10::prim::ListUnpack) {
    // Recurse through unpacks until we find a host_side_cast or otherwise
    // return false
    for (const torch::jit::Value *output : n->outputs()) {
      if (output->uses().size() != 1) {
        continue;
      }
      if (isBeforeHostSideCast(output->uses()[0].user)) {
        return true;
      }
    }
  }

  // Otherwise, the presence or lack of a host_side_cast will indicate whether
  // to return true or false
  return n->kind() == symbols::poptorch::host_side_cast;
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
      node->kind() == c10::aten::reshape || node->kind() == c10::aten::roll ||
      node->kind() == c10::aten::select || node->kind() == c10::aten::slice ||
      node->kind() == c10::aten::split || node->kind() == c10::aten::stack ||
      node->kind() == c10::aten::squeeze ||
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

  static std::unordered_set<std::string> warned_types;
  if (warned_types.find(unsupported_type) == warned_types.end()) {
    logging::warn(
        "{}: {} is not supported natively on IPU, loss of "
        "range/precision may occur. We will only warn on the first instance.",
        nodeToString(node), unsupported_type);
    warned_types.insert(unsupported_type);
  }
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
    auto *input = node->input(dtype_index)->node();

    if (node->input(dtype_index)->uses().size() == 1) {
      // Type constant is only used once, change its value
      input->i_(c10::attr::value, replacement);
    } else {
      // Create a new constant as the constant is used elsewhere
      auto no_inputs = [](torch::jit::Value *value) {
        ERROR("A constant should have no inputs");
        return value; // ensures correct output type
      };
      auto *new_type_const = node->owningGraph()->createClone(input, no_inputs);
      new_type_const->i_(c10::attr::value, replacement);
      node->replaceInput(dtype_index, new_type_const->output());
      new_type_const->insertBefore(node);
    }

    logging::info("Replacing cast to {} with cast to {} for {}",
                  c10::toString(unsupported_dtype),
                  c10::toString(replacement_dtype), nodeToString(node));
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
  logging::LogContext ctx_func("CheckAndChangeOutputTypes");
  for (auto *n : graph->nodes()) {
    // Some unpacks will happen before the host side cast, so ignore them here
    if (isBeforeHostSideCast(n)) {
      continue;
    }

    logging::LogContext ctx("processing " + nodeToString(n));

    for (auto *output : n->outputs()) {
      logging::LogContext ctx_2(output->debugName());

      checkAndChangeOutputTypesForOutput(n, output);
    }
  }
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
