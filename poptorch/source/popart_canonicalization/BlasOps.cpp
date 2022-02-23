// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace {

torch::jit::Node *matmulHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::mm(Tensor self, Tensor mat2) -> (Tensor)
  // "aten::matmul(Tensor self, int dim) -> Tensor"
  // We will fuse the batch dimesion of the matrix A of matmul(A, B),
  // if we find such a pattern:
  //
  //    matrix A(N, M, K) multiplies matrix B(K, L)
  //
  // where matrix A is matmul's input 0, and matrix B is its input 1.
  // The matrix A will be reshaped into A(N*M, K) before matmul,
  // The benefit of this transformation is to avoid the ReduceSum
  // of the backwrad pass, as ReduceSum is a performance bottleneck otherwise.
  //
  // The input IR before canonicalization:
  // %output : Float(3:14, 2:7, 7:1) = aten::matmul(%input.1, %27)
  // It takes 3 steps for the transformation:
  // 1. Reshape
  // 2. Matmul
  // 3. Reshape
  // The output IRs after canonicalization:
  // %28 : Float(6:7, 7:1) =
  //       popart::reshape_static_shape[shape=[6, 7]](%input.1)
  // %29 : FloatTensor = popart::matmul(%28, %27)
  // %30 : Float(3:14, 2:7, 7:1) =
  //       popart::reshape_static_shape[shape=[3, 2, 7]](%29)

  torch::jit::Value *matrix_a = node->input(0);
  torch::jit::Value *matrix_b = node->input(1);

  std::vector<std::int64_t> shape_input_a = shapeFromTensor(matrix_a);
  std::vector<std::int64_t> shape_input_b = shapeFromTensor(matrix_b);
  std::int64_t size_a = shape_input_a.size();
  std::int64_t size_b = shape_input_b.size();

  torch::jit::Node *result;
  // Matrix A can have any batch dimensions
  // But matrix B has to be in a 2D shape
  if (size_a >= 3 && size_b == 2 &&
      shape_input_a[size_a - 1] == shape_input_b[0]) {
    // Prepare the output shape of matmul by
    //   - merging all the batch dimensions of matrix A, and
    //   - taking the last dimension of matrix B
    std::vector<std::int64_t> output_shape;
    // Prepare the shape of fused batch dimensions for matrix A
    std::vector<std::int64_t> fused_a_shape;

    std::int64_t merged_dim = shape_input_a[size_a - 2];
    for (std::int64_t i = 0; i < size_a - 2; ++i) {
      // Final output shape could have any batch dimensions as before
      output_shape.push_back(shape_input_a[i]);
      merged_dim *= shape_input_a[i];
    }
    output_shape.push_back(shape_input_a[size_a - 2]);
    output_shape.push_back(shape_input_b[size_b - 1]);
    // Matrix A has 2D shape after fusing batch dimensions
    fused_a_shape.push_back(merged_dim);
    fused_a_shape.push_back(shape_input_a[size_a - 1]);

    // 1. Reshape matrix A to merge all of its batch size dimensions
    torch::jit::Node *merge_mat = createReshape(graph, matrix_a, fused_a_shape);
    // 2. Matmul
    torch::jit::Node *mul =
        createMatmul(graph, {merge_mat->output(), matrix_b});
    // 3. Reshape to the expected shape of the original matmul
    result = createReshape(graph, mul->output(), output_shape);
    // Add the trace to ease debugging for before and after IRs
    logging::trace("Replacing matmul {} with {} {} {}", *node, *merge_mat, *mul,
                   *result);

  } else {
    // The "normal" matmul will follow the original path
    result = createMatmul(graph, {matrix_a, matrix_b});
  }
  return result;
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::matmul, matmulHandler);

  // Matrix-Vector
  registerHandler(c10::aten::mv, matmulHandler);

  // Vector-Vector
  registerHandler(c10::aten::dot, matmulHandler);

  // With bias.
  registerHandler(c10::aten::bmm, matmulHandler);

  // No bias.
  registerHandler(c10::aten::mm, matmulHandler);
}

} // namespace poptorch
