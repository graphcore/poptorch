// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TYPE_INFERENCE_HANDLER_HPP_
#define POPTORCH_DISPATCH_TYPE_INFERENCE_HANDLER_HPP_

#include <functional>
#include <optional>
#include <unordered_map>

#include <ATen/Tensor.h>
#include <ATen/core/boxing/KernelFunction.h>

#include "Tensor.hpp"

namespace poptorch {

class TypeInferenceHandler {
public:
  explicit TypeInferenceHandler(TensorStore *tensor_store)
      : _tensor_store(tensor_store) {}

  void inferOutputTypes(const c10::OperatorHandle &op, c10::Stack *ipu_stack);

  struct Workaround {
    std::function<bool(std::size_t, const c10::IValue &, const c10::Stack &)>
        predicate_fn;
    std::function<c10::IValue(const c10::IValue &, const c10::Stack &)>
        transform_fn;
  };

private:
  // Create a meta tensor with the same type as the input
  static at::Tensor toMeta(const at::Tensor &tensor, bool upcast_to_long,
                           bool is_prelu = false);

  // Create a stack of meta tensors that matches the inputs in
  // ipu_stack
  static c10::Stack createMetaStack(const c10::Stack &ipu_stack,
                                    const std::string &schema_key,
                                    bool is_prelu);

  // Using the computed meta output stack, repopulate the ipu stack
  // with tensors of the correct inferred output types
  void repopulateIpuStack(c10::Stack *ipu_stack, const c10::Stack &meta_stack,
                          bool is_prelu);

  static c10::optional<std::size_t>
  indexArgToUpcast(const std::string &schema_key);

  static std::optional<Workaround>
  workaroundLookup(const std::string &schema_key);
  static c10::IValue applyWorkaround(const Workaround &workaround,
                                     std::size_t value_index,
                                     const c10::IValue &value,
                                     const c10::Stack &stack);
  static const std::unordered_map<std::string, Workaround> schema_to_workaround;

  TensorStore *_tensor_store;
};
} // namespace poptorch

#endif // POPTORCH_DISPATCH_TYPE_INFERENCE_HANDLER_HPP_
