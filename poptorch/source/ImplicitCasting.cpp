// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

HalfFloatCasting &getHalfFloatCastingBehavior() {
  static HalfFloatCasting behavior = HalfFloatCasting::Default;

  return behavior;
}

bool &getRunningStatisticsAlwaysFloat() {
  static bool always_float = true;

  return always_float;
}

} // namespace

void setHalfFloatCastingBehavior(const HalfFloatCasting behavior) {
  getHalfFloatCastingBehavior() = behavior;
}

HalfFloatCasting halfFloatCastingBehavior() {
  return getHalfFloatCastingBehavior();
}

void setRunningStatisticsAlwaysFloat(bool value) {
  logging::debug("poptorch.Options set runningStatisticsAlwaysFloat to {}",
                 value);
  getRunningStatisticsAlwaysFloat() = value;
}

bool runningStatisticsAlwaysFloat() {
  return getRunningStatisticsAlwaysFloat();
}

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

c10::ScalarType promoteTypes(c10::ScalarType t1, c10::ScalarType t2) {
  ERROR_ON_MSG(isCompilingWithDispatcher(),
               "promoteTypes() shouldn't be called in the dispatcher");
  auto hf_behavior = halfFloatCastingBehavior();
  if (hf_behavior == HalfFloatCasting::FloatDowncastToHalf ||
      hf_behavior == HalfFloatCasting::Default) {
    if ((t1 == c10::ScalarType::Half && t2 == c10::ScalarType::Float) ||
        (t1 == c10::ScalarType::Float && t2 == c10::ScalarType::Half)) {
      return c10::ScalarType::Half;
    }
  } else {
    ERROR_ON(hf_behavior != HalfFloatCasting::HalfUpcastToFloat);
  }

  auto type = c10::promoteTypes(t1, t2);

  // Ensure we don't promote to a type the IPU doesn't support
  type = coerceToSupportedType(type);

  return type;
}

c10::ScalarType highestTypeOf(const std::vector<c10::ScalarType> &types) {
  ERROR_ON_MSG(types.empty(),
               "Unsupported node: could not identify scalar or tensor inputs.");

  auto new_type = types[0];
  for (size_t i = 1; i < types.size(); i++) {
    // If using a "half or float" with a specified type, resolve it here
    // except for bool which is even lower in the priority and an int type
    if (types[i] == HALF_OR_FLOAT) {
      if (new_type == c10::ScalarType::Bool || !c10::isFloatingType(new_type)) {
        new_type = HALF_OR_FLOAT;
      }
      continue;
    }

    if (new_type == HALF_OR_FLOAT) {
      if (types[i] == c10::ScalarType::Bool || !c10::isFloatingType(new_type)) {
        continue;
      }

      new_type = types[i];
    }

    new_type = poptorch::promoteTypes(new_type, types[i]);
  }

  return new_type;
}

c10::ScalarType inferExpectedTypeDispatch(
    const torch::jit::ArrayRef<torch::jit::Value *> &inputs,
    const ImplicitCast implicit_cast) {
  // Work out the types of all inputs
  at::native::ResultTypeState state = {};

  unsigned int input_num = 0;
  for (auto *input : inputs) {
    logging::LogContext ctx(std::string("processing input ") +
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

c10::ScalarType
inferExpectedType(const torch::jit::ArrayRef<torch::jit::Value *> &inputs,
                  const ImplicitCast implicit_cast) {
  // Work out the types of all inputs
  std::vector<at::ScalarType> tensor_types;
  std::vector<at::ScalarType> number_types;

  unsigned int input_num = 0;
  for (auto *input : inputs) {
    logging::LogContext ctx(std::string("processing input ") +
                            std::to_string(input_num));

    if (!skipInput(implicit_cast, input_num) &&
        input->type()->kind() != c10::TypeKind::NoneType) {
      auto *node = input->node();
      auto tensor_type = input->type()->expect<c10::TensorType>();
      ERROR_ON(!tensor_type->scalarType());

      // Normally the scalar type will be emplaced in the back of tensor_types.
      // But if it was originally a python numeric type, the implicit casting
      // rules change such that it is ignored with two exceptions below.
      std::vector<at::ScalarType> *place_in = &tensor_types;

      if (node->kind() == symbols::poptorch::tensor_constant) {
        if (getNodeTensorAttrValue(node)
                .unsafeGetTensorImpl()
                ->is_wrapped_number()) {
          place_in = &number_types;
        }
      }

      place_in->emplace_back(*tensor_type->scalarType());
    }
    input_num++;
  }

  // Exception 1
  // There may only be wrapped numbers, so python casting rules should apply.
  // (In trace, this should not occur.)
  if (tensor_types.empty()) {
    return highestTypeOf(number_types);
  }

  auto highest_type = highestTypeOf(tensor_types);

  // Exception 2
  // If the number is floating point type and the current
  // highest is not, it always becomes a float32
  // (torch.tensor([1, 2], dtype=torch.bfloat16)*2.0).dtype == torch.bfloat16
  // (torch.tensor([1, 2], dtype=torch.int16)*2.0).dtype == torch.float32
  // (torch.tensor([1, 2], dtype=torch.int64)*2.0).dtype ==torch.float32

  if (!c10::isFloatingType(highest_type)) {
    for (auto type : number_types) {
      if (c10::isFloatingType(type)) {
        return at::ScalarType::Float;
      }
    }
  }

  // General case
  return highest_type;
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
  c10::ScalarType expected_type;
  // The dispatcher version of mixed-precision type inference simply delegates
  // to PyTorch's own routines, so that we always match their decisions.
  // We have legacy behavior in tracing which requires a lot of complicated
  // computation for type resolution, so we split here. All the legacy code can
  // be removed when tracing is removed.
  if (isCompilingWithDispatcher()) {
    expected_type = inferExpectedTypeDispatch(*inputs, implicit_cast);
  } else {
    expected_type = inferExpectedType(*inputs, implicit_cast);
  }

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
      ERROR_ON(constant->kind() != symbols::poptorch::tensor_constant);
      to_delete.push_back(constant);
    }
  }

  for (auto *node : to_delete) {
    node->destroy();
  }
}

} // namespace poptorch
