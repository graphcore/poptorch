// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <vector>

#include "poptorch_logging/Logging.hpp"

#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {
namespace {

const char *fail_msg = "The size of the sliced tensor must be a constant for "
                       "each execution of the model when running on the IPU.";

// Extract the constant used in the supplied add/subtract node and increase or
// decrease size accordingly. Negate reverses the sign.
void extractAddSubtractConstant(torch::jit::Node *node, std::int64_t *size,
                                bool negate) {
  ERROR_ON_MSG(node->kind() != symbols::popart::add &&
                   node->kind() != symbols::popart::sub,
               fail_msg);
  ERROR_ON(node->inputs().size() != 2);

  auto *constant = isAnyConstant(node->input(0)->node())
                       ? node->input(0)->node()
                       : node->input(1)->node();

  if (node->kind() == symbols::popart::sub) {
    negate = !negate;
  }

  if (isFloatingPointConstant(constant)) {
    ERROR(fail_msg << " In this case, there is a float added to the slice "
                   << "indices meaning it may change between runs.");
  }

  if (negate) {
    (*size) -= constantToLong(constant);
  } else {
    (*size) += constantToLong(constant);
  }
}

// Returns the input of a node which is not a constant, if any. Otherwise,
// returns null. Raises an error if there are more than one such input.
torch::jit::Node *getOnlyNonConstantInput(torch::jit::Node *node) {
  torch::jit::Node *only_such_input = nullptr;

  for (auto *input : node->inputs()) {
    if (!isAnyConstant(input->node())) {
      if (only_such_input != nullptr) {
        logging::trace("dynamicSliceHandler failed due to a node with multiple "
                       "non constant inputs when seeking a shared ancestor "
                       "node. Offending node: {}",
                       *node);
        ERROR(fail_msg);
      }
      only_such_input = input->node();
    }
  }

  return only_such_input;
}

// Returns true if the nodes always yield the same output.
bool nodesAlwaysSameOutput(torch::jit::Node *a, torch::jit::Node *b) {
  // Check same kind
  if (a->kind() != b->kind()) {
    return false;
  }

  // Avoid random nodes
  if (isNondeterministic(*a) || isNondeterministic(*b)) {
    return false;
  }

  // Check same inputs
  if (a->inputs().size() != b->inputs().size()) {
    return false;
  }

  const auto *a_it = a->inputs().begin();
  const auto *b_it = b->inputs().begin();
  for (; a_it != a->inputs().end(); a_it++, b_it++) {
    if (!nodesAlwaysSameOutput((*a_it)->node(), (*b_it)->node())) {
      return false;
    }
  }

  // Check same attributes
  if (a->numAttributes() != b->numAttributes()) {
    return false;
  }

  auto a_attributes_names = a->attributeNames();
  for (auto attrib_name : a_attributes_names) {
    if (!attributeEqual(a, b, attrib_name)) {
      return false;
    }
  }

  return true;
}

// Convert any inputs to the specified node which are a cast of another
// constant into a single (already cast) constant
void resolveCastConstants(torch::jit::Graph *graph, torch::jit::Node *node) {
  for (auto *input : node->inputs()) {
    // Move on if it is not a cast situation
    auto *cast_node = input->node();
    if (cast_node->kind() != symbols::popart::cast) {
      continue;
    }

    auto *constant_to_be_cast = cast_node->input()->node();

    if (constant_to_be_cast->kind() != symbols::poptorch::tensor_constant) {
      continue;
    }

    // Obtain the tensor and cast
    auto tensor = constant_to_be_cast->t(c10::attr::value);
    auto popart_cast_to = cast_node->s(c10::Symbol::attr("to"));
    auto scalar_type = onnxStrToScalarType(popart_cast_to.c_str());
    tensor.to(scalar_type);

    // Replace node to avoid a cast
    torch::jit::WithInsertPoint insert_point(node);
    auto *replacement_node = tensorToConstant(graph, tensor);
    replaceAllUsesWith(cast_node->output(), replacement_node->output());

    markNodeForDeletion(cast_node);
    markNodeForDeletion(constant_to_be_cast);
  }
}

// Follow the inputs of each node until we reach a common ancestor.
// Every node in the chain must only have one non-constant input for this to
// work. If this were not the case, the setup is unlikely to resolve to a case
// in which dynamic slice could work (exceptions include adding an input
// multiplied by zero, etc). Therefore, this limitation is not an issue in
// practice.
void populateAncestory(torch::jit::Graph *graph,
                       std::vector<torch::jit::Node *> *start_ancestory,
                       std::vector<torch::jit::Node *> *end_ancestory,
                       torch::jit::Node *start_node,
                       torch::jit::Node *end_node) {
  torch::jit::Node *start_ancestor = start_node;
  torch::jit::Node *end_ancestor = end_node;

  while (start_ancestor != end_ancestor) {
    // Push back whichever node is later
    bool end_is_later = end_ancestor->isAfter(start_ancestor);

    auto **later_node = end_is_later ? &end_ancestor : &start_ancestor;
    auto *add_to_list = end_is_later ? end_ancestory : start_ancestory;
    add_to_list->push_back(*later_node);

    // The algorithm will fail if there is an input that would be a constant but
    // for a cast. The best solution is to cast the constant to elimate the
    // cast.
    resolveCastConstants(graph, *later_node);

    // Update either start_ancestor or end_ancestor by going a step along the
    // chain of non-constant inputs
    *later_node = getOnlyNonConstantInput(*later_node);

    if (*later_node == nullptr) {
      logging::trace("dynamicSliceHandler failed due to lack of a shared "
                     "ancestor.");
      ERROR(fail_msg);
    }
  }

  // Do a sanity check and log the results to a trace
  ERROR_ON(start_ancestor == nullptr);
  logging::trace("Shared ancestor: {}\n", *start_ancestor);
  logging::trace("Start ancestory:");
  for (auto it = start_ancestory->rbegin(); it != start_ancestory->rend();
       it++) {
    logging::trace("{}", **it);
  }
  logging::trace("End ancestory:");
  for (auto it = end_ancestory->rbegin(); it != end_ancestory->rend(); it++) {
    logging::trace("{}", **it);
  }
}

// Remove nodes which are common across the start of both node ancestries
void removeCommonNodes(std::vector<torch::jit::Node *> *start_ancestory,
                       std::vector<torch::jit::Node *> *end_ancestory) {
  while (!(start_ancestory->empty() || end_ancestory->empty())) {
    if (nodesAlwaysSameOutput(start_ancestory->back(), end_ancestory->back())) {
      start_ancestory->pop_back();
      end_ancestory->pop_back();
    } else {
      break;
    }
  }

  if (start_ancestory->empty() && end_ancestory->empty()) {
    ERROR("The start and end of a slice must be different.");
  }
}

// Obtain the size of the slice based on the processed start/end ancestory.
// This involves processing add and subtract nodes and their constants.
std::int64_t
determineSizeConstant(const std::vector<torch::jit::Node *> &start_ancestory,
                      const std::vector<torch::jit::Node *> &end_ancestory) {
  std::int64_t size = 0;

  for (auto *node : start_ancestory) {
    if (node->kind() == c10::aten::Int ||
        node->kind() == symbols::popart::cast) {
      continue;
    }

    extractAddSubtractConstant(node, &size, true);
  }

  for (auto *node : end_ancestory) {
    if (node->kind() == c10::aten::Int ||
        node->kind() == symbols::popart::cast) {
      continue;
    }

    extractAddSubtractConstant(node, &size, false);
  }

  logging::trace("Size determined to be: {}", size);
  return size;
}

// Handle a slice in which the start is an arbitary (i.e. non constant) input
// but the slice is a fixed size
torch::jit::Node *dynamicSliceHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node,
                                      torch::jit::Node *start_node,
                                      torch::jit::Node *end_node) {
  std::vector<torch::jit::Node *> start_ancestory;
  std::vector<torch::jit::Node *> end_ancestory;

  // Obtain the path from the nodes back to a common node
  populateAncestory(graph, &start_ancestory, &end_ancestory, start_node,
                    end_node);

  // Remove any common nodes at the beginning of each ancestory
  // NB this is used in finding the size of the slice only and does not affect
  // the start node.
  removeCommonNodes(&start_ancestory, &end_ancestory);

  // Calculate the size of the slice
  std::int64_t size = determineSizeConstant(start_ancestory, end_ancestory);

  // The == 0 case should be taken care of already but having it here stops
  // lint errors for dividing by 0.
  if (size <= 0) {
    ERROR("Taking a slice of a tensor with the end less than the start is "
          "not supported.");
  }

  // The dim is as usual
  std::int64_t dim = constantToLong(node->input(1)->node());

  auto length_of_dim = shapeFromTensor(node->input(0))[dim];

  ERROR_ON_MSG(length_of_dim % size != 0,
               "The size of the slice ("
                   << size << ") must be a factor of the slicing "
                   << "dimension (" << length_of_dim << ").");

  // Make sure the start_node is a tensor not an int
  if (start_node->output()->type()->kind() == c10::TypeKind::IntType) {
    start_node = start_node->input()->node();
  }

  // Reshape the start node from a scalar to a one-dim and cast to UINT32
  start_node = createReshape(graph, start_node->output(), {1});
  start_node = createCast(graph, {start_node->output()}, "UINT32");

  auto *new_node = createDynamicslice(
      graph, {node->input(0), start_node->output()}, {dim}, {size},
      1); // No overlap 1 assumed
  return new_node;
}

// implements slicing with step by subsampling a slice with unit step
torch::jit::Node *subsampleSlice(torch::jit::Graph *graph,
                                 torch::jit::Node *slice, int dims, int dim,
                                 int step) {
  if (step != 1) {
    std::vector<int64_t> strides(dims, static_cast<int64_t>(1));
    strides[dim] = step;
    slice = createSubsample(graph, {slice->output()}, strides);
  }

  return slice;
}

namespace {
torch::jit::Node *sliceCommon(torch::jit::Graph *graph, torch::jit::Node *node,
                              torch::jit::Value *input, int64_t dim,
                              torch::jit::Node *start_node,
                              torch::jit::Node *end_node, int64_t step) {
  auto dims = shapeFromTensor(input);
  if (dim < 0) {
    dim += dims.size();
  }

  // If any of the inputs are not constants, dynamicSlice is required
  if (!isTensorConstant(start_node) || !isTensorConstant(end_node)) {
    auto *slice = dynamicSliceHandler(graph, node, start_node, end_node);
    return subsampleSlice(graph, slice, dims.size(), dim, step);
  }

  std::int64_t start = constantToLong(start_node);
  std::int64_t end = constantToLong(end_node);

  // If we slice a scalar we should do nothing.
  if (dims.empty()) {
    return createIdentity(graph, {input});
  }

  // Based on aten/src/ATen/native/TensorShape.cpp slice()
  if (start < 0) {
    start += dims[dim];
  }
  if (end < 0) {
    end += dims[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= dims[dim]) {
    start = dims[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= dims[dim]) {
    end = dims[dim];
  }

  auto *slice = createSlice(graph, {input}, {end}, {start}, {dim});
  return subsampleSlice(graph, slice, dims.size(), dim, step);
}
} // namespace

torch::jit::Node *sliceHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
  auto *input = node->input(0);
  auto dim = constantToLong(node->input(1)->node());
  auto *start_node = node->input(2)->node();
  auto *end_node = node->input(3)->node();
  auto *step_node = node->input(4)->node();

  ERROR_ON_MSG(!isTensorConstant(step_node), "Slicing step must be a constant");

  auto step = constantToLong(step_node);
  ERROR_ON_MSG(step < 1, "Slicing step must be at least 1");

  return sliceCommon(graph, node, input, dim, start_node, end_node, step);
}

torch::jit::Node *unbindHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::unbind(Tensor self, int dim) -> Tensor[]

  auto *x = node->input(0);
  auto shape = shapeFromTensor(x);
  int dim = constantToInt(node->input(1)->node());
  std::int64_t dim_size = shape[dim];

  std::vector<torch::jit::Value *> tensors;
  // Select each index in dimension 'dim' of x and add all
  // slices to a vector
  for (std::int64_t i = 0; i < dim_size; i++) {
    auto *inds = wrapInConstant1D(graph, i);
    auto *gather = createGather(graph, {x, inds}, dim);
    // Squeeze out the gathered dim
    auto *squeeze = createSqueeze(graph, {gather->output()}, {dim});
    tensors.push_back(squeeze->output());
  }

  return createAndInsertNode(graph, at::prim::ListConstruct, tensors);
}

torch::jit::Node *narrowHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
  auto *input = node->input(0);
  int dim = constantToInt(node->input(1)->node());
  auto *start_node = node->input(2)->node();
  auto *end_node = node->input(3)->node();

  return sliceCommon(graph, node, input, dim, start_node, end_node, 1);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::slice, sliceHandler);
  registerHandler(c10::aten::unbind, unbindHandler);
  registerHandler(c10::aten::narrow, narrowHandler);
}

} // namespace poptorch
