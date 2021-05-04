// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include <poptorch/OpBuilder.hpp>
#include <poptorch/Utils.hpp>
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

namespace poptorch {
namespace {

torch::jit::Node *bilinearHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias)
  // -> Tensor

  // Bilinear - outputs a linear combination of feature inputs:
  //
  //     Ynm = \sum_ij Un_i Am_ij Vn_j + bm
  //
  // Where U and V are the data input tensors containing feature vectors
  // (possibly ND), A is the 3D weight tensor, and b is the bias vector.
  // We can evaluate the bilinear map in pytorch as follows:
  //
  //     U = U.unsqueeze(-2).unsqueeze(-2)
  //     V = V.unsqueeze(-2).unsqueeze(-1)
  //     Y = U.matmul(A).matmul(V)
  //     Y = Y.squeeze(-1).squeeze(-1)
  //     Y = Y + b

  // Tensor feature inputs
  torch::jit::Value *in1 = node->input(0);
  torch::jit::Value *in2 = node->input(1);

  // weight and the optional bias
  torch::jit::Value *weight = node->input(2);
  torch::jit::Value *bias = node->input(3);

  // Insert singleton dimensions in feature inputs
  auto shape1 = shapeFromTensor(in1);
  shape1.insert(shape1.end() - 1, 1);
  shape1.insert(shape1.end() - 1, 1);
  torch::jit::Node *flat_in1 = createReshape(graph, in1, shape1);

  auto shape2 = shapeFromTensor(in2);
  shape2.insert(shape2.end() - 1, 1);
  shape2.insert(shape2.end(), 1);
  torch::jit::Node *flat_in2 = createReshape(graph, in2, shape2);

  // Multiply matrices together for the bilinear map: U * A * V as above
  torch::jit::Node *in1_matmul_weight =
      poptorch::createMatmul(graph, {flat_in1->output(), weight});

  torch::jit::Node *bilinear_map = poptorch::createMatmul(
      graph, {in1_matmul_weight->output(), flat_in2->output()});

  // Squeeze out the trailing singleton dims by reshaping to the expected
  // result size. Taking care to omit the singleton dims injected above, we
  // derive the output shape from the leading dimensions of input1 and the
  // size in the first dimension of the weight tensor.  In pytorch:
  //
  //    U.shape[0:-1] + (A.shape[0],)
  //
  // is the expected output size.
  std::vector<std::int64_t> result_shape(shape1.begin(), shape1.end() - 3);
  auto weight_shape = shapeFromTensor(weight);
  result_shape.push_back(weight_shape.front());
  torch::jit::Node *result =
      createReshape(graph, bilinear_map->output(), result_shape);

  // Add optional bias
  if (!isNone(bias->node())) {
    result = poptorch::createAdd(graph, {result->output(), bias});
  }

  return result;
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::bilinear, bilinearHandler);
}

} // namespace poptorch
