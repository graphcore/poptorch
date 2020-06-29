// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PoptorchSymbols.h"
#include <poptorch/OpBuilder.hpp>
#include <poptorch_logging/Error.hpp>
namespace poptorch {

/*
 * Manually added operation.
 */
torch::jit::Node *CreateReshape(torch::jit::Graph &graph, torch::jit::Value *A,
                                const std::vector<int64_t> &new_shape) {
  torch::jit::Node *newNode =
      graph.create(Symbols::popart::reshape_static_shape, {A});
  newNode->is_(c10::attr::shape, new_shape);
  return newNode;
}

torch::jit::Node *Create_ConstantInt(torch::jit::Graph &graph,
                                     const std::vector<int64_t> &data,
                                     const std::vector<int64_t> &new_shape) {
  torch::jit::Node *newNode = graph.create(Symbols::poptorch::int_constant);
  newNode->is_(c10::attr::data, data);
  newNode->is_(c10::attr::shape, new_shape);
  return newNode;
}

torch::jit::Node *Create_ConstantFloat(torch::jit::Graph &graph,
                                       const std::vector<double> &data,
                                       const std::vector<int64_t> &new_shape) {
  torch::jit::Node *newNode = graph.create(Symbols::poptorch::float_constant);
  newNode->fs_(c10::attr::data, data);
  newNode->is_(c10::attr::shape, new_shape);
  return newNode;
}

torch::jit::Node *Create_Cast(torch::jit::Graph &graph, torch::jit::Value *A,
                              c10::ScalarType scalar) {
  torch::jit::Node *newNode = graph.create(Symbols::poptorch::cast, {A});

  std::string newType = "";

  if (scalar == c10::kFloat) {
    newType = "FLOAT";
  } else if (scalar == c10::kInt) {
    newType = "INT32";
  } else if (scalar == c10::kLong) {
    newType = "INT64";
  }

  newNode->s_(c10::Symbol::fromQualString("attr::type"), newType);

  return newNode;
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

torch::jit::Node *Create_ConstantPad(torch::jit::Graph &graph,
                                     torch::jit::Value *A,
                                     const std::vector<int64_t> &pad_shape,
                                     float constant) {
  torch::jit::Node *newNode =
      graph.create(Symbols::poptorch::constant_pad, {A});
  newNode->is_(c10::Symbol::fromQualString("attr::pads"),
               convertPytorchPads(pad_shape));
  newNode->f_(c10::Symbol::fromQualString("attr::value"), constant);
  return newNode;
}

torch::jit::Node *Create_EdgePad(torch::jit::Graph &graph, torch::jit::Value *A,
                                 const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *newNode = graph.create(Symbols::poptorch::edge_pad, {A});
  newNode->is_(c10::Symbol::fromQualString("attr::pads"),
               convertPytorchPads(pad_shape));
  return newNode;
}

torch::jit::Node *Create_ReflectionPad(torch::jit::Graph &graph,
                                       torch::jit::Value *A,
                                       const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *newNode =
      graph.create(Symbols::poptorch::reflection_pad, {A});
  newNode->is_(c10::Symbol::fromQualString("attr::pads"),
               convertPytorchPads(pad_shape));

  return newNode;
}

torch::jit::Node *Create_addNotInPlace(torch::jit::Graph &graph,
                                       torch::jit::Value *A,
                                       torch::jit::Value *B) {
  torch::jit::Node *newNode =
      graph.create(Symbols::poptorch::addNotInPlace, {A, B});
  return newNode;
}

/*
 * Auto generated operation.
 */
#include "CompilerOps.cpp.inc"

} // namespace poptorch
