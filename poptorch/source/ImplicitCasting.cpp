// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/native/TypeProperties.h>
#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <vector>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/ImplicitCasting.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {

namespace {
bool skipInput(const ImplicitCast implicit_cast, const unsigned int input_num) {
  ERROR_ON(implicit_cast == ImplicitCast::None);

  if (implicit_cast == ImplicitCast::ExceptFirst && input_num == 0) {
    return true;
  }
  if (implicit_cast == ImplicitCast::ExceptSecond && input_num == 1) {
    return true;
  }
  if (implicit_cast == ImplicitCast::ExceptThird && input_num == 2) {
    return true;
  }
  if (implicit_cast == ImplicitCast::ExceptFourthFifth &&
      (input_num == 3 || input_num == 4)) {
    return true;
  }

  return false;
}

c10::ScalarType inferExpectedTypeDispatch(
    const torch::jit::ArrayRef<torch::jit::Value *> &inputs,
    const ImplicitCast implicit_cast) {
  // Work out the types of all inputs
  at::native::ResultTypeState state = {};

  unsigned int input_num = 0;
  for (auto *input : inputs) {
    logging::LogContext const ctx(std::string("processing input ") +
                                  std::to_string(input_num));

    if (!skipInput(implicit_cast, input_num) &&
        input->type()->kind() != c10::TypeKind::NoneType) {
      auto tensor_type = input->type()->expect<c10::TensorType>();
      ERROR_ON(!tensor_type->scalarType());

      auto osizes = tensor_type->sizes().concrete_sizes();
      std::vector<int64_t> sizes;
      if (osizes) {
        sizes = *osizes;
      }
      state = at::native::update_result_type_state(
          at::native::empty_cpu(c10::IntArrayRef(sizes.data(), sizes.size()),
                                tensor_type->scalarType()),
          state);
    }
    input_num++;
  }

  return at::native::result_type(state);
}

bool needToRetype(const torch::jit::Value *input,
                  const c10::ScalarType expected_type) {
  if (input->type()->kind() == c10::TypeKind::NoneType) {
    return false;
  }

  ERROR_ON(input->node()->kind() == at::prim::Constant);

  auto input_type = input->type()->cast<c10::TensorType>()->scalarType();
  return input_type != expected_type;
}

torch::jit::Value *addCast(torch::jit::Value *input,
                           const c10::ScalarType type) {
  torch::jit::Node *node = input->node();
  auto *new_node = createCast(input->owningGraph(), input, type);
  auto current_type = input->type()->cast<c10::TensorType>();

  new_node->output()->setType(current_type->withScalarType(type));
  node->replaceInputWith(input, new_node->output());

  return new_node->output();
}

} // namespace

std::vector<torch::jit::Value *>
implicitCastInputs(torch::jit::ArrayRef<torch::jit::Value *> *inputs,
                   const ImplicitCast implicit_cast) {
  // The dispatcher version of mixed-precision type inference simply delegates
  // to PyTorch's own routines, so that we always match their decisions.
  c10::ScalarType const expected_type =
      inferExpectedTypeDispatch(*inputs, implicit_cast);

  std::vector<torch::jit::Value *> new_inputs;

  unsigned int input_num = 0;
  for (auto *input : *inputs) {
    if (!skipInput(implicit_cast, input_num) &&
        needToRetype(input, expected_type)) {
      new_inputs.push_back(addCast(input, expected_type));
    } else {
      new_inputs.push_back(input);
    }
    input_num++;
  }
  return new_inputs;
}

void removeDeadImplicitCasts(torch::jit::Graph *graph) {
  // We are removing dead code casts that result from the following cases:
  //   - Torch is dispatching a cast of a tensor in which case it should be used
  //     elsewhere and its uses won't be empty -> just delete the cast.
  //   - Torch is dispatching a cast of a wrapped number (a tensor_constant on
  //     our side) -> delete the cast and the constant.
  std::vector<torch::jit::Node *> to_delete;

  for (auto *node : graph->nodes()) {
    if (node->kind() != symbols::popart::cast || node->hasUses()) {
      continue;
    }

    to_delete.push_back(node);
    if (node->input()->uses().size() == 1) {
      // 'node' is the only use so it's safe to delete. This must be a
      // tensor_constant representing a wrapped number.
      auto *constant = node->input()->node();
      if (constant->kind() == symbols::poptorch::tensor_constant) {
        to_delete.push_back(constant);
      }
    }
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

} // namespace poptorch
