// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>

#include "poptorch_logging/Error.hpp"

#include "../CompilerHelpers.hpp"

namespace poptorch_ir {

void lstm_input::lowerToPoplar(CompilerContext &context) {
  std::vector<poplar::Tensor> hidden_layers = context.fromSsa(this->hx());
  std::vector<poplar::Tensor> params = context.fromSsa(this->params());

  ERROR_ON(!this->has_biases());
  ERROR_ON(this->num_layers() != 1);
  ERROR_ON(this->dropout().convertToFloat() != 0.0);

  poplar::Tensor raw_input = context.fromSsa(this->input());
  // Poplibs expects order [sequence_length, batch_size, input_size].
  if (this->batch_first()) {
    raw_input = raw_input.dimShuffle({1, 0, 2});
  }

  // An LSTM state is made of 4 values
  constexpr std::uint64_t state_size = 4;
  ERROR_ON(params[0].shape()[0] % 4 != 0);
  const std::uint64_t hidden_size = params[0].shape()[0] / state_size;

  // Prepare the weights

  // Use this function to obtain a view on the poplibs tensors suitable for
  // copying PyTorch tensors
  auto pytorch_view_of = [&hidden_size](const poplar::Tensor &poplibs_tensor,
                                        bool are_weights) {
    // Poplibs needs weights to be shape [state_size, x_size, hidden_size]
    // and biases to be shape [state_size, hidden_size]
    // where x_size is either input_size or hidden_size

    // PyTorch weights are shape [state_size*hidden_size, x_size]
    // Pytorch biases are shape [state_size*hidden_size]

    // Perform the slice and shuffle here. Note that this is the inverse, i.e.
    // take the poplibs weight and transform it into the PyTorch weight.

    // Both input-hidden and hidden-hidden weights need to be transposed the
    // same way, removing the ambiguity of hidden_size used for both dims.

    std::vector<poplar::Interval> intervals{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    std::vector<poplar::Tensor> slices = poplibs_tensor.slices(intervals, 0);

    // Transpose h_size and hidden_size if weights (not biases)
    if (are_weights) {
      for (size_t i = 0; i < slices.size(); i++) {
        slices[i] = slices[i].dimShuffle({0, 2, 1});
      }
    }

    // Poplibs needs order FICO while PyTorch uses order IFCO
    // i - input gate o - output gate f - forget gate c - cell gate
    poplar::Tensor wf = slices[0];
    poplar::Tensor wi = slices[1];
    poplar::Tensor wc = slices[2];
    poplar::Tensor wo = slices[3];

    poplar::Tensor concat = poplar::concat({wi, wf, wc, wo}, 0);

    poplar::Tensor reshaped;
    if (are_weights) {
      reshaped =
          concat.reshape({state_size * hidden_size,
                          concat.numElements() / (state_size * hidden_size)});
    } else {
      reshaped = concat.reshape({state_size * hidden_size});
    }

    return reshaped;
  };

  const popnn::lstm::LstmParams lstm_params(
      raw_input.elementType(), raw_input.shape()[1], raw_input.shape()[0],
      {raw_input.shape()[2], hidden_size});

  // Create poplib weights and copy the pytorch weights
  popnn::lstm::LstmWeights weights = popnn::lstm::createWeights(
      context.graph, lstm_params, poplar::DebugContext("LstmWeights"));
  context.seq.add(poplar::program::Copy(
      params[0], pytorch_view_of(weights.inputWeights, true)));
  context.seq.add(poplar::program::Copy(
      params[1], pytorch_view_of(weights.outputWeights, true)));

  // Poplibs LSTM takes a sum of the respective biases
  poplar::Tensor pytorch_biases =
      popops::add(context.graph, params[2], params[3], context.seq);
  context.seq.add(poplar::program::Copy(
      pytorch_biases, pytorch_view_of(weights.biases, false)));

  // Handle initial state

  // Ignore matmul planning cache and options for now
  popnn::lstm::LstmState lstm_state = popnn::lstm::createInitialState(
      context.graph, lstm_params, poplar::DebugContext("LstmInitialState"));

  ERROR_ON(hidden_layers.size() != 2);
  poplar::Tensor init_h = lstm_state.output;
  context.seq.add(poplar::program::Copy(hidden_layers[0], init_h, false));

  poplar::Tensor init_c = lstm_state.cellState;
  context.seq.add(poplar::program::Copy(hidden_layers[1], init_c, false));

  // Always copy input for now
  poplar::Tensor input = context.graph.clone(raw_input);
  context.seq.add(poplar::program::Copy(raw_input, input));

  poplar::Tensor output;
  poplar::Tensor cell_state;

  // TODO(T68091) support set_available_memory

  std::tie(output, cell_state) = popnn::lstm::lstmFwd(
      context.graph, lstm_params, lstm_state, input, weights, nullptr,
      context.seq, poplar::DebugContext("LstmFwd"));

  // The Poplibs output will be [sequence_length, batch_size, input_size]
  // This is correct for Pytorch except if batch_first is specified.
  if (this->batch_first()) {
    context.addTensor(this->output(), output.dimShuffle({1, 0, 2}));
  } else {
    context.addTensor(this->output(), output);
  }

  // h_n needs to be size [1, batch_size, hidden_size]
  // (1 as we support only 1 layer and no bidirectional)
  // For one layer, this is simply the output at the last time step.
  poplar::Tensor output_h_state =
      output.slice(lstm_params.timeSteps - 1, lstm_params.timeSteps, 0);
  context.addTensor(
      this->hy(),
      output_h_state.reshape(
          {1, static_cast<std::uint64_t>(lstm_params.batchSize), hidden_size}));

  // c_n also needs the same resize
  context.addTensor(
      this->cy(),
      cell_state.reshape(
          {1, static_cast<std::uint64_t>(lstm_params.batchSize), hidden_size}));
}

} // namespace poptorch_ir
