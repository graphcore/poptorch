
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *lstmHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::lstm(Tensor self, Tensor[] hx, Tensor[] weights, bool bias,
  // int num_layers, float dropout, bool training, bool bidirectional,
  // bool batch_first) -> Tensor, (Tensor, Tensor)

  torch::jit::Value *input = node->input(0);

  torch::jit::ArrayRef<torch::jit::Value *> hidden_layers =
      node->input(1)->node()->inputs();
  torch::jit::ArrayRef<torch::jit::Value *> weights_list =
      node->input(2)->node()->inputs();

  bool use_bias = constantToBool(node->input(3)->node());
  ERROR_ON_MSG(!use_bias, "LSTM without biases not supported");
  std::int64_t num_layers = constantToLong(node->input(4)->node());

  ERROR_ON_MSG(num_layers != 1, "Only LSTM with 1 layer supported");

  float dropout = constantToFloat(node->input(5)->node());
  ERROR_ON_MSG(dropout != 0.0f, "LSTM only supports dropout = 0.0");

  bool bidirectional = constantToBool(node->input(7)->node());
  ERROR_ON_MSG(bidirectional, "bidirectional LSTM not supported");

  bool batch_first = constantToBool(node->input(8)->node());

  // An LSTM state is made of 4 values
  constexpr std::uint64_t state_size = 4;
  const std::int64_t num_weights =
      *weights_list[0]->type()->expect<c10::TensorType>()->sizes()[0];
  ERROR_ON(num_weights % state_size != 0);
  const std::int64_t num_hidden_layers = num_weights / state_size;

  // def reshape_weights(onnx_weights):
  //    ws = builder.aiOnnx.split([w], 4, 1, [hidden_size] * 4)
  //    ws = [builder.aiOnnx.transpose([i], [0, 2, 1]) for i in ws]
  //    ws = builder.aiOnnx.concat([ws[i] for i in (2, 0, 3, 1)], 0)
  //    return ws
  //
  // Note: onnx weights are in IOFC order while Torch uses IFCO
  //
  // Biases don't need to be transposed
  auto reshape_tensor = [&](torch::jit::Value *values, bool areWeights) {
    const std::uint64_t num_dims_without_batch = areWeights ? 2 : 1;
    std::vector<std::int64_t> shape = shapeFromTensor(values);
    if (shape.size() == num_dims_without_batch) {
      // Add a batch dimension
      shape.insert(shape.begin(), 1);
      torch::jit::Node *reshape = createReshape(graph, values, shape);
      values = reshape->output();
    }
    torch::jit::Node *states =
        createSplit(graph, {values}, state_size, 1,
                    {num_hidden_layers, num_hidden_layers, num_hidden_layers,
                     num_hidden_layers});
    std::vector<torch::jit::Value *> slices;
    for (std::uint64_t i = 0; i < state_size; ++i) {
      if (areWeights) {
        // Weights also need to be transposed
        torch::jit::Node *transposed =
            createTranspose(graph, {states->output(i)}, {0, 2, 1});
        slices.push_back(transposed->output());
      } else {
        slices.push_back(states->output(i));
      }
    }
    torch::jit::Node *concat =
        createConcat(graph, {slices[1], slices[0], slices[2], slices[3]}, 0);
    return concat->output();
  };

  torch::jit::Node *concat_weights =
      createConcat(graph,
                   {reshape_tensor(weights_list[0], true),
                    reshape_tensor(weights_list[1], true)},
                   1);
  torch::jit::Node *combine_biases =
      createAddNotInPlace(graph, reshape_tensor(weights_list[2], false),
                          reshape_tensor(weights_list[3], false));

  torch::jit::Node *concat_states =
      createConcat(graph, {hidden_layers[0], hidden_layers[1]}, 0);

  std::vector<std::int64_t> input_shape = shapeFromTensor(input);
  std::int64_t batch_dim = 0;
  // Transpose output BSF -> SBF
  if (batch_first) {
    torch::jit::Node *transpose = createTranspose(graph, {input}, {1, 0, 2});
    input = transpose->output();
    batch_dim = 1;
  }
  std::vector<torch::jit::Value *> args;
  args.push_back(input);
  args.push_back(concat_weights->output()); // input weights + output_weights
  args.push_back(combine_biases->output()); // biases
  args.push_back(concat_states->output());  // init_states

  torch::jit::Node *lstm = createLstm(graph, args, 1);

  // Keep the last slice from Y `[seq_length, num_directions, batch_size,
  // hidden_size]
  torch::jit::Node *y_h = createSlice(
      graph,
      {lstm->output(0), wrapInConstant1D(graph, input_shape[batch_dim] - 1),
       wrapInConstant1D(graph, INT_MAX), wrapInConstant1D(graph, 0)});

  torch::jit::Value *output = lstm->output(0);
  // Transpose output SBF -> BSF
  if (batch_first) {
    torch::jit::Node *transpose = createTranspose(graph, {output}, {1, 0, 2});
    output = transpose->output();
  }

  ERROR_ON(node->outputs().size() != 3);
  if (node->hasUses()) {
    replaceOutputUse(node->output(0), output);
    replaceOutputUse(node->output(1), y_h->output());
    replaceOutputUse(node->output(2), lstm->output(1));
  }

  markNodeForDeletion(node);
  return nullptr;
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::lstm, lstmHandler);
// clang-format on

} // namespace poptorch
