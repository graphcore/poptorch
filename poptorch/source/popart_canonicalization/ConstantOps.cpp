// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <random>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace {

torch::jit::Node *onesZerosHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool?
  //            pin_memory) -> Tensor
  // aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool?
  //             pin_memory) -> Tensor
  // aten::zeros_like(Tensor self, ScalarType? dtype, Layout? layout, Device?
  //                  device, bool? pin_memory, MemoryFormat? memory_format)
  //                  -> Tensor
  // aten::ones_like(Tensor self, ScalarType? dtype, Layout? layout, Device?
  //                 device, bool? pin_memory, MemoryFormat? memory_format)
  //                 -> Tensor

  torch::jit::Symbol kind = node->kind();
  const bool is_ones = kind == c10::aten::ones ||
                       kind == c10::aten::ones_like ||
                       kind == c10::aten::new_ones;

  auto *output = node->output();
  auto *new_node = createAndInsertNode(
      graph, is_ones ? symbols::poptorch::ones : symbols::poptorch::zeros, {},
      ImplicitCast::None, OutputType::AsDtype, 1, getNodeScalarType(output));

  if (kind != c10::aten::new_ones && kind != c10::aten::new_zeros) {
    new_node->is_(c10::attr::shape, shapeFromTensor(output));
  } else {
    const auto shape_list = handleTensorList(node->input(1)->node());
    std::vector<int64_t> shape;
    for (auto *size : shape_list) {
      ERROR_ON_MSG(
          !isTensorConstant(size->node()),
          "Invalid shape for "
          "new_zeros or new_ones. Shape needs to be a static constant");
      shape.emplace_back(constantToInt(size->node()));
    }
    new_node->is_(c10::attr::shape, shape);
  }
  return new_node;
}

torch::jit::Node *arangeHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  std::size_t start;
  std::size_t end;
  std::size_t step;

  switch (node->inputs().size()) {
  // arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
  // arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None,
  //        Device? device=None, bool? pin_memory=None) -> Tensor
  case 2:
  case 5:
    start = 0;
    end = constantToLong(node->input(0)->node());
    step = 1;
    break;
  // arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None,
  //              Layout? layout=None, Device? device=None,
  //              bool? pin_memory=None) -> Tensor
  case 6:
    start = constantToLong(node->input(0)->node());
    end = constantToLong(node->input(1)->node());
    step = 1;
    break;
  // arange.start_out(Scalar start, Scalar end, Scalar step=1, *,
  //                  Tensor(a!) out) -> Tensor(a!)
  // arange.start_step(Scalar start, Scalar end, Scalar step, *,
  //                   ScalarType? dtype=None, Layout? layout=None,
  //                   Device? device=None, bool? pin_memory=None) -> Tensor
  case 4:
  case 7:
    start = constantToLong(node->input(0)->node());
    end = constantToLong(node->input(1)->node());
    step = constantToLong(node->input(2)->node());
    break;
  default:
    ERROR("Unsupported arange op");
    break;
  }

  std::vector<std::int64_t> vals((end - start) / step);
  size_t v = start;
  std::generate(std::begin(vals), std::end(vals), [&v, step] {
    auto cv = v;
    v += step;
    return cv;
  });

  return createConstantInt(graph, vals,
                           {static_cast<std::int64_t>(vals.size())});
}

torch::jit::Node *randpermHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::randperm(Scalar n, ScalarType dtype, Layout, Device, bool pin_memory)
  auto *n = node->input(0)->node();
  setNodeTensorAttrValue(n, getNodeTensorAttrValue(n).to(at::ScalarType::Long));
  n->output()->inferTypeFrom(getNodeTensorAttrValue(n));
  auto *size_of_permutation = n->output();

  const std::vector<int64_t> shape = {constantToLong(n)};
  const auto dtype = c10::ScalarType::Float;

  torch::jit::Value *uniform =
      createRandomUniform(graph, nullptr, shape, 1.0, 0.0, dtype)->output();

  auto *topk = createTopk(graph, {uniform, size_of_permutation}, 0,
                          true /*largest*/, true /*sorted*/);

  return createCast(graph, topk->output(1), c10::ScalarType::Int);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::arange, arangeHandler);
  registerHandler(c10::aten::ones, onesZerosHandler);
  registerHandler(c10::aten::ones_like, onesZerosHandler);
  registerHandler(c10::aten::new_ones, onesZerosHandler);
  registerHandler(c10::aten::new_zeros, onesZerosHandler);
  registerHandler(c10::aten::zeros, onesZerosHandler);
  registerHandler(c10::aten::zeros_like, onesZerosHandler);
  registerHandler(c10::aten::randperm, randpermHandler);
}

} // namespace poptorch
