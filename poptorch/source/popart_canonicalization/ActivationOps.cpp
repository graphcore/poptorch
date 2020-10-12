// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
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
} // namespace

// clang-format off
static bool handlers =
    registerHandlers(
        c10::aten::glu, gluHandler);
// clang-format on

} // namespace poptorch
