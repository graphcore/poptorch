// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {

torch::jit::Node *celuHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // CELU(x) = max(x, 0) + min(0, a * (exp(x/a)-1))
  auto x = node->input(0);
  auto a = node->input(1);
  auto zero = createConstantFloatLike(graph, x, {0.0}, {})->output();

  auto max = createMax(graph, {x, zero})->output();
  auto div = createDiv(graph, {x, a})->output();
  auto expm1 = createExpm1(graph, {div})->output();
  auto mul = createMul(graph, {a, expm1})->output();
  auto min = createMin(graph, {zero, mul})->output();

  return createAdd(graph, {max, min});
}

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

torch::jit::Node *hardshrinkHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // hardshrink(x) = x, if x > lambda
  //               = x, if x < -lambda
  //               = 0, otherwise
  torch::jit::Value *x = node->input(0);
  torch::jit::Value *lambda = node->input(1);
  torch::jit::Value *neg_lambda = createNeg(graph, {lambda})->output();

  torch::jit::Value *x_gt_lambda = createGreater(graph, {x, lambda})->output();
  torch::jit::Value *x_lt_neg_lambda =
      createLess(graph, {x, neg_lambda})->output();

  torch::jit::Value *mask =
      createLogical_or(graph, {x_gt_lambda, x_lt_neg_lambda})->output();

  torch::jit::Value *zero =
      createConstantFloatLike(graph, x, {0.0}, {})->output();
  return createWhere(graph, {mask, x, zero});
}

torch::jit::Node *rreluHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // training: rrelu(x)  = x if x >= 0
  //                     = a * x if x < 0, where a uniformly random value
  //                                       from [lower, upper]
  // inference: rrelu(x) = x if x >= 0
  //                     = x * ((lower + upper) / 2)
  auto x = node->input(0);
  auto lower = constantToFloat(node->input(1)->node());
  auto upper = constantToFloat(node->input(2)->node());
  auto is_training = constantToInt(node->input(3)->node());

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
  torch::jit::Value *x = node->input(0);
  torch::jit::Value *beta = node->input(1);
  torch::jit::Value *threshold = node->input(2);

  // softplus = 1/beta * log(1 + exp(beta * x))
  torch::jit::Value *beta_x = createMul(graph, {x, beta})->output();
  torch::jit::Value *exp_betax = createExp(graph, {beta_x})->output();
  torch::jit::Value *log1p_exp = createLog1p(graph, {exp_betax})->output();
  torch::jit::Value *softplus = createDiv(graph, {log1p_exp, beta})->output();

  // For numerical stability, revert to identity when beta * x > threshold
  torch::jit::Value *mask = createGreater(graph, {beta_x, threshold})->output();
  return createWhere(graph, {mask, x, softplus});
}

torch::jit::Node *softshrinkHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // softshrink(x) = x - lambda, if x > lambda
  //               = x + lambda, if x < -lambda
  //               =          0, otherwise
  torch::jit::Value *x = node->input(0);
  torch::jit::Value *lambda = node->input(1);
  torch::jit::Value *neg_lambda = createNeg(graph, {lambda})->output();

  torch::jit::Value *x_plus_lambda = createAdd(graph, {x, lambda})->output();
  torch::jit::Value *x_minus_lambda = createSub(graph, {x, lambda})->output();
  torch::jit::Value *zero =
      createConstantFloatLike(graph, x, {0.0}, {})->output();

  torch::jit::Value *x_gt_lambda = createGreater(graph, {x, lambda})->output();
  torch::jit::Value *shrink =
      createWhere(graph, {x_gt_lambda, x_minus_lambda, zero})->output();

  torch::jit::Value *x_lt_neg_lambda =
      createLess(graph, {x, neg_lambda})->output();

  return createWhere(graph, {x_lt_neg_lambda, x_plus_lambda, shrink});
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::celu, celuHandler);
  registerHandler(c10::aten::glu, gluHandler);
  registerHandler(c10::aten::hardshrink, hardshrinkHandler);
  registerHandler(c10::aten::rrelu, rreluHandler);
  registerHandler(c10::aten::softplus, softplusHandler);
  registerHandler(c10::aten::softshrink, softshrinkHandler);
}

} // namespace poptorch
