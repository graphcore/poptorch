// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <any>
#include <iterator>
#include <limits>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {
namespace {

bool isHostSideNode(torch::jit::Node *n) {
  std::vector<torch::jit::Node *> to_check;
  to_check.push_back(n);
  while (!to_check.empty()) {
    auto cur_node = to_check.back();
    to_check.pop_back();
    for (auto output : cur_node->outputs()) {
      for (auto use : output->uses()) {
        auto use_kind = use.user->kind();
        if (use_kind != c10::prim::Return &&
            use_kind != symbols::poptorch::set_available_memory &&
            use_kind != symbols::poptorch::set_matmul_serialization &&
            use_kind != c10::prim::TupleConstruct &&
            use_kind != c10::prim::ListConstruct &&
            use_kind != c10::prim::TupleUnpack &&
            use_kind != c10::prim::ListUnpack) {
          return false;
        }
        to_check.push_back(use.user);
      }
    }
  }
  return true;
}

void replaceWithConstantTensor(torch::jit::Graph *graph, torch::jit::Node *n,
                               const at::Tensor &t) {
  torch::jit::WithInsertPoint insert_point(n);
  auto new_node = tensorToConstant(graph, t, isHostSideNode(n));

  // Due to tracing ambiguity, a float tensor here could be either float or half
  auto new_type = new_node->output()->type()->expect<c10::TensorType>();
  if (new_type->scalarType() == at::ScalarType::Float) {
    new_node->output()->setType(new_type->withScalarType(HALF_OR_FLOAT));
  }

  for (size_t use_idx = 0; use_idx < n->output()->uses().size(); use_idx++) {
    auto u = n->output()->uses()[use_idx];
    u.user->replaceInput(u.offset, new_node->output());
    use_idx--;
  }
}

void warnDoubleOutOfRange(double val, torch::jit::Node *n) {
  if (val > std::numeric_limits<float>::max() ||
      val < std::numeric_limits<float>::lowest()) {
    logging::warn("{}: torch.float64 constant cannot be represented as a "
                  "torch.float32",
                  nodeToString(n));
  }
}

void warnLongOutOfRange(int64_t val, torch::jit::Node *n) {
  if (val > std::numeric_limits<int32_t>::max() ||
      val < std::numeric_limits<int32_t>::lowest()) {
    logging::warn("{}: torch.int64 constant cannot be represented as a "
                  "torch.int32",
                  nodeToString(n));
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
      warnDoubleOutOfRange(s.toDouble(), n);
      dtype = at::kFloat;
    } else if (s.isIntegral(false)) {
      dtype = at::kInt;

      // Handle magic number 9223372036854775807
      if (s.toLong() == std::numeric_limits<int64_t>::max()) {
        s = std::numeric_limits<int32_t>::max();
        logging::info("{}: Using max value for torch.int32 in place of max "
                      "value for torch.int64",
                      nodeToString(n));
      } else {
        warnLongOutOfRange(s.toLong(), n);
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
  ERROR_ON(!tensor.defined());
  bool was_wrapped = tensor.unsafeGetTensorImpl()->is_wrapped_number();
  if (tensor.scalar_type() == at::ScalarType::Double) {
    warnDoubleOutOfRange(
        *reinterpret_cast<double *>(tensor.unsafeGetTensorImpl()->data()), n);
    tensor = tensor.to(at::ScalarType::Float);
  }
  if (tensor.scalar_type() == at::ScalarType::Long) {
    warnLongOutOfRange(
        *reinterpret_cast<int64_t *>(tensor.unsafeGetTensorImpl()->data()), n);

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

void handleList(torch::jit::Graph *graph, torch::jit::Node *n,
                std::unordered_set<torch::jit::Node *> *to_delete) {
  torch::jit::WithInsertPoint insert_point(n);

  // Turn each element into a prim constant
  auto value_vec = n->ival(c10::attr::value).toListRef();
  std::vector<torch::jit::Node *> prim_consts;
  for (auto &val : value_vec) {
    prim_consts.emplace_back(graph->create(c10::prim::Constant));

    if (n->output()->type()->isSubtypeOf(c10::ListType::ofBools())) {
      prim_consts.back()->i_(c10::attr::value, val.toBool());
      prim_consts.back()->output()->setType(c10::BoolType::get());
    } else if (n->output()->type()->isSubtypeOf(c10::ListType::ofFloats())) {
      prim_consts.back()->f_(c10::attr::value, val.toDouble());
      prim_consts.back()->output()->setType(c10::FloatType::get());
    } else if (n->output()->type()->isSubtypeOf(c10::ListType::ofInts())) {
      prim_consts.back()->i_(c10::attr::value, val.toInt());
      prim_consts.back()->output()->setType(c10::IntType::get());
    } else if (n->output()->type()->isSubtypeOf(c10::ListType::ofTensors()) ||
               n->output()->type()->isSubtypeOf(c10::ListType::create(
                   c10::OptionalType::create(c10::TensorType::get())))) {
      if (!val.isNone()) {
        // Treat node as a regular tensor
        auto tensor = val.toTensor();
        prim_consts.back()->t_(c10::attr::value, tensor);
        prim_consts.back()->output()->inferTypeFrom(tensor);
      } else {
        // Assign NoneType so that the node can be skipped over
        // during constant canonicalization
        prim_consts.back()->output()->setType(c10::NoneType::get());
      }
    } else {
      ERROR("Unexpected type");
    }

    graph->insertNode(prim_consts.back());
  }

  // Add a list construct
  auto list_construct = graph->create(c10::prim::ListConstruct);
  for (auto prim_const : prim_consts) {
    list_construct->addInput(prim_const->output());
  }

  graph->insertNode(list_construct);
  n->output()->replaceAllUsesWith(list_construct->output());

  // Canonicalize each constant individually and ensure deletion
  for (auto prim_const : prim_consts) {
    // If there are NoneTypes we can skip those
    if (prim_const->output()->type() != c10::NoneType::get()) {
      if (prim_const->output()->type()->isSubtypeOf(c10::TensorType::get())) {
        handleTensorConstant(graph, prim_const);
      } else {
        handleNumberConstant(graph, prim_const);
      }
      to_delete->insert(prim_const);
    }
  }
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
    } else if (node->output()->type()->isSubtypeOf(c10::ListType::ofBools())) {
      // Only known case is the result of an evaluated constexpr
      logging::LogContext ctx2("handling as bool list constant");
      handleList(graph, node, &to_delete);
    } else if (node->output()->type()->isSubtypeOf(c10::ListType::ofFloats())) {
      // Only known case is the result of an evaluated constexpr
      logging::LogContext ctx2("handling as float list constant");
      handleList(graph, node, &to_delete);
    } else if (node->output()->type()->isSubtypeOf(c10::ListType::ofInts())) {
      // Only known case is the result of an evaluated constexpr
      logging::LogContext ctx2("handling as int list constant");
      handleList(graph, node, &to_delete);
    } else if (node->output()->type()->isSubtypeOf(
                   c10::ListType::ofTensors())) {
      // Only known case is the result of an evaluated constexpr
      logging::LogContext ctx2("handling a tensor list constant");
      handleList(graph, node, &to_delete);
    } else if (node->output()->type()->isSubtypeOf(c10::ListType::create(
                   c10::OptionalType::create(c10::TensorType::get())))) {
      logging::LogContext ctx2("handling an optional tensor list constant");
      handleList(graph, node, &to_delete);
    } else {
      ERROR("Unsupported type " << node->output()->type()->str());
    }

    to_delete.insert(*it);
  }
  searchAndPossiblyDestroy(to_delete);
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
