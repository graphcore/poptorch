// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "poptorch_logging/Error.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {
torch::jit::Node *
createAndInsertNode(torch::jit::Graph *graph, torch::jit::NodeKind kind,
                    torch::jit::ArrayRef<torch::jit::Value *> inputs,
                    size_t num_outputs) {
  torch::jit::Node *new_node = graph->create(kind, inputs, num_outputs);

  graph->insertNode(new_node);
  return new_node;
}

torch::jit::Node *tensorToConstant(torch::jit::Graph *graph,
                                   const at::Tensor &t) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::tensor_constant);
  new_node->output()->inferTypeFrom(t);
  new_node->t_(c10::attr::value, t);

  return new_node;
}

/*
 * Manually added operation.
 */
torch::jit::Node *createReshape(torch::jit::Graph *graph, torch::jit::Value *A,
                                const std::vector<int64_t> &new_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::popart::reshape_static_shape, {A});
  new_node->is_(c10::attr::shape, new_shape);
  return new_node;
}

torch::jit::Node *createConstantInt(torch::jit::Graph *graph,
                                    const std::vector<int64_t> &data,
                                    const std::vector<int64_t> &new_shape) {
  auto total_size = static_cast<size_t>(std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<int64_t>()));

  size_t stride = 0;
  if (data.size() != 1) {
    ERROR_ON(total_size != data.size());
    stride = 1;
  }

  auto t =
      at::empty({new_shape}, at::dtype(at::ScalarType::Int)
                                 .memory_format(c10::MemoryFormat::Contiguous));

  for (size_t i = 0; i < total_size; i++) {
    // NOLINTNEXTLINE
    *(t.data_ptr<int32_t>() + i) = static_cast<int32_t>(data[i * stride]);
  }

  // Use bounds checking for the last element
  data.at((total_size - 1) * stride);

  return tensorToConstant(graph, t);
}

torch::jit::Node *createConstantFloat(torch::jit::Graph *graph,
                                      const std::vector<double> &data,
                                      const std::vector<int64_t> &new_shape) {
  auto total_size = static_cast<size_t>(std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<std::int64_t>()));
  size_t stride = 0;
  if (data.size() != 1) {
    ERROR_ON(total_size != data.size());
    stride = 1;
  }

  auto t =
      at::empty({new_shape}, at::dtype(at::ScalarType::Float)
                                 .memory_format(c10::MemoryFormat::Contiguous));

  for (size_t i = 0; i < total_size; i++) {
    // NOLINTNEXTLINE
    *(t.data_ptr<float>() + i) = static_cast<float>(data[i * stride]);
  }

  // Use bounds checking for the last element
  data.at((total_size - 1) * stride);

  return tensorToConstant(graph, t);
}

torch::jit::Node *createCast(torch::jit::Graph *graph, torch::jit::Value *A,
                             c10::ScalarType scalar) {
  std::string new_type = scalarTypeToOnnxString(scalar);
  return createCast(graph, {A}, new_type);
}

static std::vector<std::int64_t>
convertPytorchPads(const std::vector<int64_t> &pad_shape) {
  std::vector<int64_t> tmp = pad_shape;

  // Work out how many dimensions we are padding. Each dimension is in the form
  // (begin1, end1, beginN, endN)
  const std::size_t num_dimensions = tmp.size() / 2;

  ERROR_ON_MSG(num_dimensions > 3, "Internal error: Unsupported number of "
                                   "dimensions in constant pad operation.");

  // Move from pytorch (begin1, end1, beginN, endN) to popart (begin1, beginN,
  // end1, endN) It's also in reverse order.
  if (num_dimensions == 2) {
    // Move to popart ordering.
    std::swap(tmp[1], tmp[2]);

    // Reverse the order.
    std::swap(tmp[0], tmp[1]);
    std::swap(tmp[2], tmp[3]);

  } else if (num_dimensions == 3) {
    // Move to popart ordering and reverse.
    tmp = {
        // The begins.
        tmp[4],
        tmp[2],
        tmp[0],

        // The ends.
        tmp[5],
        tmp[3],
        tmp[1],
    };
  }
  // Padding is applying to the * dimensions in (N, C, *) but popart allows N/C
  // to be specified as well. So add 4 zeros to make it compatable. (N_beg,
  // C_beg, *_beg, N_end, C_end, *_end).
  tmp.insert(tmp.begin(), 0);
  tmp.insert(tmp.begin(), 0);

  // Insert after the first two zeros and after however many dimensions there
  // are.
  tmp.insert(tmp.begin() + num_dimensions + 2, 0);
  tmp.insert(tmp.begin() + num_dimensions + 2, 0);

  return tmp;
}

torch::jit::Node *createConstantPad(torch::jit::Graph *graph,
                                    torch::jit::Value *A,
                                    const std::vector<int64_t> &pad_shape,
                                    float constant) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::constant_pad, {A});
  new_node->is_(c10::Symbol::fromQualString("attr::pads"),
                convertPytorchPads(pad_shape));
  new_node->f_(c10::Symbol::fromQualString("attr::value"), constant);
  return new_node;
}

torch::jit::Node *createEdgePad(torch::jit::Graph *graph, torch::jit::Value *A,
                                const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::edge_pad, {A});
  new_node->is_(c10::Symbol::fromQualString("attr::pads"),
                convertPytorchPads(pad_shape));
  return new_node;
}

torch::jit::Node *createReflectionPad(torch::jit::Graph *graph,
                                      torch::jit::Value *A,
                                      const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::reflection_pad, {A});
  new_node->is_(c10::Symbol::fromQualString("attr::pads"),
                convertPytorchPads(pad_shape));

  return new_node;
}

torch::jit::Node *createAddNotInPlace(torch::jit::Graph *graph,
                                      torch::jit::Value *A,
                                      torch::jit::Value *B) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::add_not_in_place, {A, B});
  return new_node;
}

torch::jit::Node *
createCustomOperation(torch::jit::Graph *graph,
                      const std::vector<torch::jit::Value *> &inputs,
                      const std::string &name, const std::string &domain,
                      std::int64_t domainVersion, std::int64_t numOutputs) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::custom_operation, inputs, numOutputs);

  new_node->s_(c10::Symbol::fromQualString("attr::name"), name);
  new_node->s_(c10::Symbol::fromQualString("attr::domain"), domain);
  new_node->i_(c10::Symbol::fromQualString("attr::version"), domainVersion);
  new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"), numOutputs);

  return new_node;
}

torch::jit::Node *createRandomNormal(torch::jit::Graph *graph,
                                     const std::vector<int64_t> &shape,
                                     float mean, float scale) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::random_normal);
  new_node->is_(c10::attr::shape, shape);
  new_node->f_(c10::attr::mean, mean);
  new_node->f_(c10::attr::scale, scale);

  return new_node;
}

torch::jit::Node *createRandomUniform(torch::jit::Graph *graph,
                                      const std::vector<int64_t> &shape,
                                      float high, float low) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::random_uniform);
  new_node->is_(c10::attr::shape, shape);
  new_node->f_(c10::attr::high, high);
  new_node->f_(c10::attr::low, low);

  return new_node;
}

/*
 * Auto generated operation.
 */
#include "CompilerOps.cpp.inc"

} // namespace poptorch
