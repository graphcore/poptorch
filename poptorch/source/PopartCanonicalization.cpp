// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

// An odd function which returns each tensor dimension as an array, a helper for
// torch.max(tensor) and torch.min(tensor). I.E a 4D tensor will return (0, 1,
// 2, 3).
std::vector<std::int64_t>
reduceHelperDimensionCreator(torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  c10::TensorTypePtr as_tensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  std::int64_t index = 0;
  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape;
  for (auto optional_int : *dims.sizes()) {
    shape.push_back(index++);
  }
  return shape;
}

class CanonicalizeImpl {
public:
  static void run(torch::jit::Graph *graph);
};

bool hasUnityValue(torch::jit::Value *value) {
  auto tensor = value->node()->t(c10::attr::value);
  if (tensor.numel() != 1) {
    return false;
  }
  return tensor.to(at::ScalarType::Float).item<float>() == 1.0;
}

/*
 * ConvertAtenToPopart implementation.
 */

void CanonicalizeImpl::run(torch::jit::Graph *graph) {
  for (torch::jit::Node *node : graph->nodes()) {
    logging::LogContext ctx("PopartCanonicalization processing " +
                            nodeToString(node));
    torch::jit::WithInsertPoint insert_point(node);
    torch::jit::Node *new_node = nullptr;
    torch::jit::Symbol kind = node->kind();

    if (SymbolHandler handler = getHandler(node)) {
      new_node = handler(graph, node);
    }

// Handle an integer dimension attribute (this can be negative hence the special
// case)
#define HANDLE_DIM(Index) handleDimensionParam(node, Index)

#define HANDLE_LIST(Index) constantToLongVec(node->input(Index)->node())

#define HANDLE_LIST_AS_IR_CONSTANT(Index)                                      \
  intVectorToIrConstant(graph, constantToLongVec(node->input(Index)->node()))

#define HANDLE_TENSOR_LIST(Index) handleTensorList(node->input(Index)->node())
#define PARAM(Index) node->input(Index)
#define COMMA ,
#define NONE

// Returns an integer list of dimension that a tensor has. For reduce functions.
// A 5D tensor would return (0, 1, 2, 3, 4)
#define DIMENISON_LENGTH_LIST(Index)                                           \
  reduceHelperDimensionCreator(node->input(Index))

// Returns the shape of the tensor as a vector of ints.
#define TENSOR_SHAPE(Index) shapeFromTensor(node->input(Index))

// Returns the output shape of the tensor as a vector of ints.
#define OUTPUT_TENSOR_SHAPE(Index) shapeFromTensor(node->output(Index))

#define TENSOR_SHAPE_AS_IR(Index) shapeFromTensorAsIR(graph, node->input(Index))

#define GET_RETURN_TYPE getNodeScalarType(node->output())

// Check if the number of inputs is |num|. Used for overload resolution.
#define NUM_INPUTS_EQUALS(Num) node->inputs().size() == Num

// Extract a float from the constant casting if required
#define CONSTANT_TO_FLOAT(Index) constantToFloat(node->input(Index)->node())

// Extract a long from the constant by casting if required
#define CONSTANT_TO_LONG(Index) constantToLong(node->input(Index)->node())

#define CONSTANT_TO_LONG_CONSTANT(Index)                                       \
  constantToLongConstant(node->input(Index)->node())->output()

// Many binary element wise operations contained a fused "Alpha" component.
// The form of this is A (+, -) B * alpha. Most of the time this will be unity
// so can be skipped but it could be non-unity and must be handled.
// If the alpha is 1 we just ignore it otherwise we perform the alpha
// multiplication and use that. This is the macro which should be used.
#define ALPHA(ValueToMultiply, AlphaParam)                                     \
  torch::jit::Value *alphaParam = AlphaParam;                                  \
  torch::jit::Value *alphaValue = ValueToMultiply;                             \
  if (!hasUnityValue(alphaParam)) {                                            \
    auto alphaNode = createMul(graph, {alphaParam, alphaValue});               \
    alphaValue = alphaNode->output();                                          \
  }

// Create a function decl with the given call and arguments.
#define OP_CONVERTOR(AtenID, PreBuildCalls, PopartBuilder, Params)             \
  else if (kind == c10::AtenID) { /* NOLINT */                                 \
    PreBuildCalls new_node = PopartBuilder(graph, Params);                     \
  }

// Create a function decl with the given call and arguments.
#define OP_CONVERTOR_WITH_CAST(AtenID, PreBuildCalls, PopartBuilder, Params,   \
                               CastType)                                       \
  else if (kind == c10::AtenID) { /* NOLINT */                                 \
    PreBuildCalls new_node = PopartBuilder(graph, Params);                     \
    new_node = createCast(graph, new_node->output(), CastType);                \
  }

#define OP_CONVERTOR_POP(Sym, PreBuildCalls, PopartBuilder, Params)            \
  else if (kind == Sym) { /* NOLINT */                                         \
    PreBuildCalls newNode = PopartBuilder(graph, Params);                      \
  }
#include "CanonicalizationOps.h.inc"

#undef OP_CONVERTOR_POP
#undef OP_CONVERTOR_WITH_CAST
#undef OP_CONVERTOR
#undef ALPHA
#undef CONSTANT_TO_LONG_CONSTANT
#undef CONSTANT_TO_LONG
#undef CONSTANT_TO_FLOAT
#undef NUM_INPUTS_EQUALS
#undef GET_RETURN_TYPE
#undef TENSOR_SHAPE_AS_IR
#undef TENSOR_SHAPE
#undef DIMENISON_LENGTH_LIST
#undef NONE
#undef COMMA
#undef PARAM
#undef HANDLE_TENSOR_LIST
#undef HANDLE_LIST_AS_IR_CONSTANT
#undef HANDLE_LIST
#undef HANDLE_DIM

    // If we have a new node add it and replace the old use.
    if (new_node) {
      // Mark this node for deletion.
      markNodeForDeletion(node);
      ERROR_ON(node->outputs().size() != new_node->outputs().size());

      if (node->hasUses()) {
        for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
          replaceOutputUse(node, new_node, i);
        }
      }
    }
  }

  // Build a list of nodes marked for deletion.
  std::unordered_set<torch::jit::Node *> to_delete;
  for (torch::jit::Node *node : graph->nodes()) {
    if (isMarkedForDeletion(node)) {
      to_delete.insert(node);
    }
  }

  // Remove the dead nodes.
  for (torch::jit::Node *node : to_delete) {
    searchAndPossiblyDestroy(node);
  }
}

} // namespace

void canonicalize(torch::jit::Graph *graph) {
  CanonicalizeImpl converter;
  converter.run(graph);
}

} // namespace poptorch
