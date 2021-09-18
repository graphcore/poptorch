// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

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
  std::vector<std::int64_t> shape_input = shapeFromTensor(input);
  std::int64_t size = shape_input.size();

  // handle python's negative indices
  if (axis < 0) {
    axis += size;
  }

  ERROR_ON_MSG(!(axis >= 0 && axis < size),
               "The second input argument of glu is not in the legal range");

  ERROR_ON_MSG(shape_input[axis] % 2,
               "Halving dimension" << axis << "must be even");

  unsigned int half_size = static_cast<unsigned int>(shape_input[axis] / 2);

  std::vector<std::int64_t> split_sizes;
  split_sizes.push_back(half_size);
  split_sizes.push_back(half_size);

  torch::jit::Node *split = createSplit(graph, {input}, 2, axis, split_sizes);
  torch::jit::Node *sigmoid = createSigmoid(graph, {split->output(1)});

  return createMul(graph, {split->output(0), sigmoid->output()});
}

torch::jit::Node *rreluHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // training: rrelu(x)  = x if x >= 0
  //                     = a * x if x < 0, where a uniformly random value
  //                                       from [lower, upper]
  // inference: rrelu(x) = x if x >= 0
  //                     = x * ((lower + upper) / 2)
  torch::jit::Value *x = node->input(0);
  const float lower = constantToFloat(node->input(1)->node());
  const float upper = constantToFloat(node->input(2)->node());
  const bool is_training = constantToBool(node->input(3)->node());

  auto val =
      is_training
          ? createRandomUniform(graph, x, shapeFromTensor(x), upper, lower)
                ->output()
          : createConstantFloatLike(graph, x, {(lower + upper) / 2}, {})
                ->output();

  auto zero = createConstantFloatLike(graph, x, {0}, {})->output();
  auto xlt0 = createLess(graph, {x, zero})->output();
  auto mul = createMul(graph, {x, val})->output();
  return createWhere(graph, {xlt0, mul, x});
}

torch::jit::Node *softplusHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto x = node->input(0);
  auto beta = node->input(1);
  auto threshold = node->input(2);

  // beta defaults to 1
  auto is_default = constantToFloat(beta->node()) == 1.0f;

  if (!is_default) {
    x = createMul(graph, {x, beta})->output();
  }

  auto y = createSoftplus(graph, {x})->output();

  if (!is_default) {
    y = createDiv(graph, {y, beta})->output();
  }

  auto below_threshold = createLess(graph, {x, threshold})->output();
  return createWhere(graph, {below_threshold, y, node->input(0)});
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::glu, gluHandler);
  registerHandler(c10::aten::rrelu, rreluHandler);
  registerHandler(c10::aten::softplus, softplusHandler);
}

} // namespace poptorch
