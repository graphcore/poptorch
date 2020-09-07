// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <climits>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {
namespace {

void replaceWithConstantTensor(torch::jit::Graph *graph, torch::jit::Node *n,
                               const at::Tensor &t) {
  torch::jit::WithInsertPoint insert_point(n);
  auto new_node = tensorToConstant(graph, t);

  for (size_t use_idx = 0; use_idx < n->output()->uses().size(); use_idx++) {
    auto u = n->output()->uses()[use_idx];
    u.user->replaceInput(u.offset, new_node->output());
    use_idx--;
  }
}

void handleNumberConstant(torch::jit::Graph *graph, torch::jit::Node *n) {
  if (n->output()->type()->isSubtypeOf(c10::BoolType::get())) {
    replaceWithConstantTensor(
        graph, n,
        at::native::scalar_tensor(*torch::jit::constant_as<bool>(n->output()),
                                  at::device(at::kCPU).dtype(at::kInt)));
  } else {
    auto s = *torch::jit::constant_as<at::Scalar>(n->output());

    c10::ScalarType dtype;
    if (s.isFloatingPoint()) {
      dtype = at::kFloat;
    } else if (s.isIntegral(false)) {
      dtype = at::kInt;

      // Handle magic number 9223372036854775807
      if (s.toLong() == LLONG_MAX) {
        s = INT_MAX;
      }

    } else {
      ERROR("Unsupported constant type");
    }

    auto wrapped_number =
        at::native::scalar_tensor(s, at::device(at::kCPU).dtype(dtype));
    wrapped_number.unsafeGetTensorImpl()->set_wrapped_number(true);
    replaceWithConstantTensor(graph, n, wrapped_number);
  }
}

void handleTensorConstant(torch::jit::Graph *graph, torch::jit::Node *n) {
  auto tensor_type = n->output()->type()->expect<c10::TensorType>();
  if (!tensor_type->scalarType().has_value()) {
    ERROR("Tensor constant without type");
  }

  auto tensor = n->t(c10::attr::value);
  bool was_wrapped = tensor.unsafeGetTensorImpl()->is_wrapped_number();
  if (tensor.scalar_type() == at::ScalarType::Double) {
    tensor = tensor.to(at::ScalarType::Float);
  }
  if (tensor.scalar_type() == at::ScalarType::Long) {
    tensor = tensor.to(at::ScalarType::Int);
  }

  // This gets lost in conversion and changes implicit casting if not set
  // (Must use an if as set_wrapped_number(false) can cause a PyTorch internal
  // error)
  if (was_wrapped) {
    tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  }

  replaceWithConstantTensor(graph, n, tensor);
}

void handleStringConstant(torch::jit::Graph *graph, torch::jit::Node *n) {
  std::string s = n->s(c10::attr::value);
  std::vector<int64_t> shape_vec;
  shape_vec.push_back(s.length());

  auto t =
      at::empty({shape_vec}, at::dtype(at::ScalarType::Char)
                                 .memory_format(c10::MemoryFormat::Contiguous));

  std::memcpy(t.data_ptr(), s.c_str(), s.length() * sizeof(char));
  replaceWithConstantTensor(graph, n, t);
}
} // namespace

void canonicaliseConstants(torch::jit::Graph *graph) {
  auto nodes = graph->nodes();
  std::unordered_set<torch::jit::Node *> to_delete;
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;

    logging::LogContext ctx("CanonicaliseConstants processing " +
                            nodeToString(node));

    if (node->kind() == c10::aten::size) {
      // This will be made a constant in the size handler
      node->output()->setType(c10::TensorType::create(c10::ScalarType::Int,
                                                      c10::nullopt, 1, false));
    }

    // If it's not a constant or if it doesn't have a value (i.e is None) or if
    // it's a Device
    if (node->kind() != c10::prim::Constant ||
        !node->hasAttribute(c10::attr::value) ||
        node->output()->type()->isSubtypeOf(c10::DeviceObjType::get())) {
      continue;
    }

    if (node->output()->type()->isSubtypeOf(c10::NumberType::get()) ||
        node->output()->type()->isSubtypeOf(c10::BoolType::get())) {
      logging::LogContext ctx2("handling as number constant");
      handleNumberConstant(graph, node);
    } else if (node->output()->type()->isSubtypeOf(c10::TensorType::get())) {
      logging::LogContext ctx2("handling as tensor constant");
      handleTensorConstant(graph, node);
    } else if (node->output()->type()->isSubtypeOf(c10::StringType::get())) {
      logging::LogContext ctx2("handling as string constant");
      handleStringConstant(graph, node);
    } else {
      ERROR("Unsupported type " << node->output()->type()->str());
    }

    to_delete.insert(*it);
  }
  searchAndPossiblyDestroy(to_delete);
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
