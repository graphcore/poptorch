// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {

torch::jit::Node *gluHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  // "aten::glu(Tensor self, int dim) -> Tensor"
  // The input IR before canonicalization:
  // %3 : Float(2:96, 4:24, 6:4, 4:1) = aten::glu(%input, %4)

  // The output IR after canonicalization. It takes 3 steps.
  // 1. split the intput into two halves
  // %5 : FloatTensor, %6 : FloatTensor = popart::split[num_outputs=2, axis=3,
  // split=[4, 4]](%input)
  // 2. sigmoid the 2nd half
  // %7 : FloatTensor = popart::sigmoid(%6)
  // 3. multiply the 1st half and the sigmoid result
  // %8 : Float(2:96, 4:24, 6:4, 4:1) = popart::mul(%5, %7)

  // Input
  torch::jit::Value *input = node->input(0);
  std::int64_t axis = constantToLong(node->input(1)->node());
  const std::vector<std::int64_t> shape_input = shapeFromTensor(input);
  const std::int64_t size = shape_input.size();

  // handle python's negative indices
  if (axis < 0) {
    axis += size;
  }

  ERROR_ON_MSG(axis < 0 || axis >= size,
               "The second input argument of glu is not in the legal range");

  ERROR_ON_MSG(shape_input[axis] % 2,
               "Halving dimension" << axis << "must be even");

  const unsigned int half_size =
      static_cast<unsigned int>(shape_input[axis] / 2);

  const std::vector<std::int64_t> split_sizes = {half_size, half_size};

  torch::jit::Node *split = createSplit(graph, {input}, 2, axis, split_sizes);
  torch::jit::Node *sigmoid = createSigmoid(graph, {split->output(1)});

  return createMul(graph, {split->output(0), sigmoid->output()});
}

torch::jit::Node *rreluHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // clang-format off
  // aten::rrelu(Tensor self, Scalar lower=0.125,
  //             Scalar upper=0.3333333333333333,
  //             bool training=False, Generator? generator=None) -> Tensor
  // aten::rrelu_with_noise(Tensor self, Tensor noise,
  //                        Scalar lower, Scalar upper,
  //                        bool training, Generator? generator) -> Tensor
  //
  // training: rrelu(x)  = x if x >= 0
  //                     = a * x if x < 0, where a uniformly random value
  //                                       from [lower, upper]
  // inference: rrelu(x) = x if x >= 0
  //                     = x * ((lower + upper) / 2)
  // clang-format on
  torch::jit::Value *x = node->input(0);
  int64_t next_idx = 1;
  if (node->kind() == c10::aten::rrelu_with_noise) {
    next_idx++; // skip noise parameter
    logging::warn("Noise parameter not supported for aten::rrelu_with_noise");
  }
  const float lower = constantToFloat(node->input(next_idx++)->node());
  const float upper = constantToFloat(node->input(next_idx++)->node());
  const bool is_training = constantToBool(node->input(next_idx++)->node());

  auto *val =
      is_training
          ? createRandomUniform(graph, x, shapeFromTensor(x), upper, lower)
                ->output()
          : createConstantFloatLike(graph, x, {(lower + upper) / 2}, {})
                ->output();

  auto *zero = createConstantFloatLike(graph, x, {0}, {})->output();
  auto *xlt0 = createLess(graph, {x, zero})->output();
  auto *mul = createMul(graph, {x, val})->output();
  return createWhere(graph, {xlt0, mul, x});
}

torch::jit::Node *softplusHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto *x = node->input(0);
  auto input_type = getNodeScalarType(x);
  auto beta = constantToFloat(node->input(1)->node());
  auto threshold = constantToFloat(node->input(2)->node());

  const auto msg =
      fmt::format("{{\"beta\":{},\"threshold\":{}}}", beta, threshold);
  auto *output_node = createCustomOperation(graph, {x}, "TorchSoftplus",
                                            "poptorch.custom_ops", 1, 1, msg);

  output_node->output(0)->setType(c10::TensorType::create(
      input_type, c10::nullopt, c10::nullopt, c10::nullopt));
  return output_node;
}

torch::jit::Node *hardsigmoidHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  auto *x = node->input(0);
  // hardsigmoid(x, 1/6, 0.5)
  return createHardsigmoid(graph, {x}, 1.0 / 6.0, 0.5);
}

torch::jit::Node *hardswishHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  auto *x = node->input(0);
  auto *t0 = createConstantFloatLike(graph, x, {0.0}, {})->output();
  auto *t1 = createMax(graph, {x, t0})->output();
  auto *t2 = createAbs(graph, {x})->output();
  auto *t3 = createConstantFloatLike(graph, x, {3.0}, {})->output();
  auto *t4 = createGreater(graph, {t2, t3})->output();
  auto *t5 = createAdd(graph, {x, t3})->output();
  auto *t6 = createMul(graph, {x, t5})->output();
  auto *t7 = createConstantFloatLike(graph, x, {6.0}, {})->output();
  auto *t8 = createDiv(graph, {t6, t7})->output();
  // where(greater(abs(x), 3), max(x, 0), (x + 3) * x / 6.0)
  return createWhere(graph, {t4, t1, t8});
}

torch::jit::Node *preluHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto *self = node->input(0);
  auto *weight = node->input(1);

  return createPrelu(graph, self, weight);
}

torch::jit::Node *geluHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  auto *input = node->input(0);
  const auto approximate = constantToString(node->input(1)->node());

  if (approximate == "tanh") {
    return createGelu(graph, {input});
  }
  if (approximate == "none") {
    // TODO(AFS-274): Implement gelu without approx, createGelu return tanh
    // approximation.
    return createGelu(graph, {input});
  }

  ERROR("Unknown GELU approximate '" << approximate << "'");
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::glu, gluHandler);
  registerHandler(c10::aten::rrelu, rreluHandler);
  registerHandler(c10::aten::rrelu_with_noise, rreluHandler);
  registerHandler(c10::aten::softplus, softplusHandler);
  registerHandler(c10::aten::hardsigmoid, hardsigmoidHandler);
  registerHandler(c10::aten::hardswish, hardswishHandler);
  registerHandler(c10::aten::prelu, preluHandler);
  registerHandler(c10::aten::gelu, geluHandler);
}

} // namespace poptorch
