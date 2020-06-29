// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PoptorchSymbols.h"
#include <poptorch/OpBuilder.hpp>

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

torch::jit::Node *Create_ConstantPad(torch::jit::Graph &graph,
                                     torch::jit::Value *A,
                                     const std::vector<int64_t> &pad_shape,
                                     float constant) {
  std::vector<int64_t> tmp = pad_shape;

  tmp.insert(tmp.begin(), 0);
  tmp.insert(tmp.begin(), 0);
  tmp.insert(tmp.begin() + 4, 0);
  tmp.insert(tmp.begin() + 4, 0);

  torch::jit::Node *newNode =
      graph.create(Symbols::poptorch::constant_pad, {A});
  newNode->is_(c10::Symbol::fromQualString("attr::pads"), tmp);
  newNode->f_(c10::Symbol::fromQualString("attr::value"), constant);
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
