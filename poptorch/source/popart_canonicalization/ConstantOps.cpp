// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

#include "poptorch_logging/Error.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace {

torch::jit::Node *onesZerosHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool?
  // pin_memory) -> Tensor
  // aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool?
  // pin_memory) -> Tensor

  torch::jit::Symbol kind = node->kind();
  bool is_ones = kind == c10::aten::ones;
  ERROR_ON(!is_ones && kind != c10::aten::zeros);

  c10::TensorTypePtr as_tensor =
      node->outputs()[0]->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();
  std::vector<std::int64_t> operation_shape;

  for (auto optional_int : *dims.sizes()) {
    operation_shape.push_back(*optional_int);
  }

  auto new_node = createAndInsertNode(
      graph, is_ones ? symbols::poptorch::ones : symbols::poptorch::zeros, {},
      ImplicitCast::None, OutputType::AsDtype);

  new_node->is_(c10::attr::shape, shapeFromTensor(node->output()));
  new_node->s_(c10::attr::dtype,
               scalarTypeToOnnxString(*as_tensor->scalarType()));
  setNodeOutputsTypes(new_node, ImplicitCast::None, OutputType::AsDtype);

  return new_node;
}

torch::jit::Node *arangeHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
  // aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype,
  // Layout, Device, bool pin_memory)

  ERROR_ON_MSG(node->inputs().size() != 5, "Unsupported arrange op");

  std::vector<std::int64_t> vals;
  std::size_t end = constantToLong(node->input(0)->node());
  for (std::size_t start = 0; start < end; ++start) {
    vals.push_back(start);
  }

  return createConstantInt(graph, vals,
                           {static_cast<std::int64_t>(vals.size())});
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::arange, arangeHandler,
    c10::aten::ones, onesZerosHandler,
    c10::aten::zeros, onesZerosHandler);
// clang-format on

} // namespace poptorch
