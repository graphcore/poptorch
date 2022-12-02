// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TYPE_INFERENCE_HANDLER_HPP_
#define POPTORCH_DISPATCH_TYPE_INFERENCE_HANDLER_HPP_

#include <ATen/Tensor.h>
#include <ATen/core/boxing/KernelFunction.h>

#include "Tensor.hpp"

namespace poptorch {

class TypeInferenceHandler {
public:
  explicit TypeInferenceHandler(TensorStore *tensor_store)
      : _tensor_store(tensor_store) {}

  void inferOutputTypes(const c10::OperatorHandle &op, c10::Stack *ipu_stack);

private:
  // Create a meta tensor with the same type as the input
  static at::Tensor toMeta(const at::Tensor &tensor, bool upcast_to_long,
                           bool is_prelu = false);

  // Create a stack of meta tensors that matches the inputs in
  // ipu_stack
  static c10::Stack createMetaStack(const c10::Stack &ipu_stack,
                                    std::string_view schema_key, bool is_prelu);

  // Using the computed meta output stack, repopulate the ipu stack
  // with tensors of the correct inferred output types
  void repopulateIpuStack(c10::Stack *ipu_stack, const c10::Stack &meta_stack,
                          bool is_prelu);

  static constexpr c10::optional<std::size_t>
  indexArgToUpcast(std::string_view schema_key) {
    if (schema_key == "aten::argmax.out" || schema_key == "aten::argmin.out") {
      return 3;
    }
    if (schema_key == "aten::gather" || schema_key == "aten::scatter.src" ||
        schema_key == "aten::scatter_.src" ||
        schema_key == "aten::scatter.value" ||
        schema_key == "aten::scatter_add" ||
        schema_key == "aten::scatter_add_" ||
        schema_key == "aten::scatter_reduce.two" ||
        schema_key == "aten::scatter_reduce_.two") {
      return 2;
    }
    if (schema_key == "aten::index.Tensor") {
      return 1;
    }
    return c10::nullopt;
  }

  TensorStore *_tensor_store;
};
} // namespace poptorch

#endif // POPTORCH_DISPATCH_TYPE_INFERENCE_HANDLER_HPP_
