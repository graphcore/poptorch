// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Value *prependDimension(torch::jit::Graph *graph,
                                    torch::jit::Value *tensor) {
  auto shape = shapeFromTensor(tensor);
  shape.insert(shape.begin(), 1);
  return createReshape(graph, tensor, shape)->output();
}

torch::jit::Value *reshapeWeights(torch::jit::Graph *graph,
                                  torch::jit::Value *tensor,
                                  unsigned slice_size, bool transpose = false,
                                  bool swap = true) {
  std::vector<torch::jit::Value *> slices;
  unsigned num_slices = 3;

  torch::jit::Node *split = createSplit(graph, {tensor}, num_slices, 1,
                                        {slice_size, slice_size, slice_size});

  for (unsigned i = 0; i < num_slices; ++i) {
    torch::jit::Value *transposed = split->output(i);
    if (transpose) {
      transposed = createTranspose(graph, {transposed}, {0, 2, 1})->output();
    }
    slices.push_back(transposed);
  }

  if (swap) {
    std::swap(slices[0], slices[1]);
  }

  torch::jit::Node *concat = createConcat(graph, slices, 1);
  return concat->output();
}

torch::jit::Node *gruHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  auto *input = node->input(0);
  auto *hx = node->input(1);
  auto params = node->input(2)->node()->inputs();

  bool bias = constantToBool(node->input(3)->node());
  int num_layers = constantToLong(node->input(4)->node());
  float dropout = constantToFloat(node->input(5)->node());
  bool bidirectional = constantToBool(node->input(7)->node());
  bool batch_first = constantToBool(node->input(8)->node());

  ERROR_ON_MSG(num_layers != 1, "Only GRU with 1 layer supported");
  ERROR_ON_MSG(dropout != 0.0f, "GRU only supports dropout = 0.0");
  ERROR_ON_MSG(bidirectional, "bidirectional GRU not supported");

  auto *gate_weights = prependDimension(graph, params[0]);
  auto *recur_weights = prependDimension(graph, params[1]);

  auto input_shape = shapeFromTensor(input);
  unsigned seq_length = input_shape[0];
  unsigned batch_size = input_shape[1];

  auto recur_shape = shapeFromTensor(recur_weights);
  unsigned hidden_size = recur_shape[2];

  gate_weights = reshapeWeights(graph, gate_weights, hidden_size);
  recur_weights = reshapeWeights(graph, recur_weights, hidden_size);
  torch::jit::Value *biases;

  if (bias) {
    auto *gate_biases = prependDimension(graph, params[2]);
    auto *recur_biases = prependDimension(graph, params[3]);

    gate_biases = reshapeWeights(graph, gate_biases, hidden_size);
    recur_biases = reshapeWeights(graph, recur_biases, hidden_size);

    biases = createConcat(graph, {gate_biases, recur_biases}, 1)->output();
  } else {
    biases = createConstantFloatLike(graph, input, {0.}, {1, 6 * hidden_size})
                 ->output();
  }

  // TODO(T54563)
  auto *seq_lens =
      createConstantInt(graph, {seq_length}, {batch_size})->output();

  if (batch_first) {
    input = createTranspose(graph, {input}, {1, 0, 2})->output();
  }

  auto *gru = createGru(
      graph, {input, gate_weights, recur_weights, biases, seq_lens, hx});

  auto *output = createSqueeze(graph, {gru->output(0)}, {1})->output();

  if (batch_first) {
    output = createTranspose(graph, {output}, {1, 0, 2})->output();
  }

  replaceOutputUse(node->output(0), output);
  replaceOutputUse(node->output(1), gru->output(1));

  markNodeForDeletion(node);
  return nullptr;
}

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
  torch::jit::Node *y_h = createSlice(graph, {lstm->output(0)}, {INT_MAX},
                                      {input_shape[batch_dim] - 1}, {0});

  torch::jit::Value *output = lstm->output(0);
  // Transpose output SBF -> BSF
  if (batch_first) {
    torch::jit::Node *transpose = createTranspose(graph, {output}, {1, 0, 2});
    output = transpose->output();
  }

  // The shape of y_c returned by PopART has shape (batch_size, hidden_size).
  // Torch's c_n output has shape
  // (num_directions * num_layers, batch_size, hidden_size), but since we don't
  // support bidirectional or > 1 layers, this dimension is always 1 so we just
  // need to prepend a single dim
  auto *y_c = createUnsqueeze(graph, {lstm->output(1)}, {0});

  ERROR_ON(node->outputs().size() != 3);
  if (node->hasUses()) {
    replaceOutputUse(node->output(0), output);
    replaceOutputUse(node->output(1), y_h->output());
    replaceOutputUse(node->output(2), y_c->output());
  }

  markNodeForDeletion(node);
  return nullptr;
}

torch::jit::Node *rnnHandler(torch::jit::Graph *graph, torch::jit::Node *node,
                             const std::string &nonlinearity) {
  // rnn_{tanh/relu}.input(Tensor input, Tensor hx, Tensor[] params,
  // bool has_biases, int num_layers, float dropout, bool train,
  // bool bidirectional, bool batch_first) -> (Tensor, Tensor)
  torch::jit::Value *input = node->input(0);
  torch::jit::Value *hx = node->input(1);

  torch::jit::ArrayRef<torch::jit::Value *> params =
      node->input(2)->node()->inputs();
  torch::jit::Value *w_ih = params[0]; // input-hidden weights
  torch::jit::Value *w_hh = params[1]; // hidden-hidden weights
  torch::jit::Value *b_ih = params[2]; // input-hidden bias
  torch::jit::Value *b_hh = params[3]; // hidden-hidden bias

  bool has_biases = constantToBool(node->input(3)->node());
  ERROR_ON_MSG(!has_biases, "RNN without biases is not supported");

  int num_layers = constantToInt(node->input(4)->node());
  ERROR_ON_MSG(num_layers != 1, "Only RNN with 1 layer is supported");

  float dropout = constantToFloat(node->input(5)->node());
  ERROR_ON_MSG(dropout != 0.0f, "RNN only supports dropout = 0.0");

  bool bidirectional = constantToBool(node->input(7)->node());
  ERROR_ON_MSG(bidirectional, "Bidirectional RNN is not supported");

  bool batch_first = constantToBool(node->input(8)->node());

  auto input_shape = shapeFromTensor(input);
  int64_t sequence_length;
  int64_t batch_size;

  if (batch_first) {
    // N, L, H_in -> L, N, H_in
    input = createTranspose(graph, {input}, {1, 0, 2})->output();
    sequence_length = input_shape.at(1);
    batch_size = input_shape.at(0);
  } else {
    sequence_length = input_shape.at(0);
    batch_size = input_shape.at(1);
  }

  auto *b = createConcat(graph, {b_ih, b_hh}, 0)->output();
  // Fix concat result shape so that we can use prependDimension().
  auto hidden_size = shapeFromTensor(b_ih).front();
  b->setType(
      b_ih->type()->expect<c10::TensorType>()->withSizes({2 * hidden_size}));

  // TODO(T54563)
  auto *sequence_lens =
      createConstantInt(graph, {sequence_length}, {batch_size})->output();

  std::vector<torch::jit::Value *> args = {
      // [seq_length, batch_size, input_size]
      input,
      // [num_directions, hidden_size, input_size]
      prependDimension(graph, w_ih),
      // [num_directions, hidden_size, hidden_size]
      prependDimension(graph, w_hh),
      // [num_directions, 2*hidden_size]
      prependDimension(graph, b),
      // [batch_size]
      sequence_lens,
      // [num_directions, batch_size, hidden_size]
      hx,
  };

  auto *rnn = createRnn(graph, args, {nonlinearity});
  auto *output_0 = createReshape(graph, rnn->output(0),
                                 {sequence_length, batch_size, hidden_size})
                       ->output();

  if (batch_first) {
    // L, N, H_out -> N, L, H_out
    output_0 = createTranspose(graph, {output_0}, {1, 0, 2})->output();
  }

  replaceOutputUse(node->output(0), output_0);
  replaceOutputUse(node->output(1), rnn->output(1));
  markNodeForDeletion(node);

  return nullptr;
}

torch::jit::Node *rnnTanhHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  return rnnHandler(graph, node, "Tanh");
}

torch::jit::Node *rnnReluHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  return rnnHandler(graph, node, "Relu");
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::gru, gruHandler);
  registerHandler(c10::aten::lstm, lstmHandler);
  registerHandler(c10::aten::rnn_tanh, rnnTanhHandler);
  registerHandler(c10::aten::rnn_relu, rnnReluHandler);
}

} // namespace poptorch
