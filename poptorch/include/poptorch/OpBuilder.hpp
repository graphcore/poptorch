// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_OP_BUILDER_HPP
#define INCLUDE_POPTORCH_OP_BUILDER_HPP
#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "poptorch/ImplicitCasting.hpp"

#include "poptorch_logging/Error.hpp"

// Represents how the output type of the op is to be determined
enum class OutputType {
  Unknown,
  AsFirstInput,
  AsThirdInput,
  FirstAsFirstInputSecondAlwaysInt,
  AsImplicitCastPromoted,
  AsDtype,
  AsDtypeOrAsPromoted,
  AlwaysBool,
  AlwaysFloat,
  AlwaysInt,
  AlwaysUint8
};

namespace c10 {
template <class T> class optional;
} // namespace c10

namespace poptorch {

// RAII object to set / clear the current source code location
// and metadata to those attached to the provided node.
// (Useful when creating / replacing nodes in the graph).
// [Important] This is not a stack: the metadata is cleared on
// destruction.
class WithNodeMetadata {
public:
  explicit WithNodeMetadata(torch::jit::Node *node);
  ~WithNodeMetadata();
};

// Set the current source code location (i.e all the nodes created
// will appear as having been instantiated from that location).
void setCurrentPythonCodeLocation(
    const torch::jit::SourceRange &source_location);

// Set the current metadata. (All the nodes created
// will have this metadata attached to them).
void setCurrentMetadata(const std::string &metadata);

void resetCurrentSourceLocation();

torch::jit::Node *createAndInsertNode(
    torch::jit::Graph *graph, torch::jit::NodeKind kind,
    torch::jit::ArrayRef<torch::jit::Value *> inputs = {},
    ImplicitCast implicit_cast = ImplicitCast::None,
    OutputType output_type = OutputType::Unknown, size_t num_outputs = 1,
    c10::optional<at::ScalarType> dtype = c10::optional<at::ScalarType>());

// All nodes should be added to the jit graph using this function or
// insertNodeBeforeNode().
// (or indirectly by using createAndInsertNode(), insertConstant()).
// These functions will ensure the new node contains all the required metadata
// before it's added to the graph.
void insertNodeInGraph(torch::jit::Graph *graph, torch::jit::Node *new_node);

void insertNodeBeforeNode(torch::jit::Node *new_node,
                          torch::jit::Node *insert_point);

void setSourceRangeToCurrentLocation(torch::jit::Node *node);

// Called by createAndInsertNode except in the cases of OutputType::AsDtype and
// OutputType::AsDtypeOrFirstInput where it should be called manually once the
// dtype attribute is set
void setNodeOutputsTypes(torch::jit::Node *node, ImplicitCast implicit_cast,
                         OutputType output_type);

enum class UseOfNode { HostSideOnly, PopARTOnly, HostSideAndPopART };

torch::jit::Value *insertConstant(torch::jit::Graph *graph,
                                  const torch::jit::IValue &val);

// Create a poptorch::tensor_constant, poptorch::host_side_tensor_constant
// or poptorch::host_and_ipu_side_tensor_constant node from the given tensors,
// setting the output type accordingly.
// A constant which is simply returned, perhaps as a tuple or list, is labelled
// as a host side constant to prevent it being placed in PopART. A constant
// which is both returned unchanged and used in PopART needs a further pass to
// split it into two constants.
torch::jit::Node *
tensorToConstant(torch::jit::Graph *graph, const at::Tensor &t,
                 UseOfNode constant_use = UseOfNode::PopARTOnly);

// Manually added.
torch::jit::Node *createReshape(torch::jit::Graph *graph, torch::jit::Value *A,
                                const std::vector<int64_t> &new_shape);

torch::jit::Node *createConstantInt(torch::jit::Graph *graph,
                                    const std::vector<std::int64_t> &data,
                                    const std::vector<std::int64_t> &new_shape);

torch::jit::Node *
createConstantFloat32(torch::jit::Graph *graph, const std::vector<double> &data,
                      const std::vector<std::int64_t> &new_shape);

// Create a constant float that inherits its underlying type (float16/32) from
// tensor t
torch::jit::Node *
createConstantFloatLike(torch::jit::Graph *graph, torch::jit::Value *t,
                        const std::vector<double> &data,
                        const std::vector<std::int64_t> &new_shape);

template <typename SymbolHandler>
torch::jit::Node *
createHandlerOperation(torch::jit::Graph *graph, SymbolHandler &&handler,
                       torch::jit::ArrayRef<torch::jit::Value *> inputs) {
  torch::jit::Node *inputs_node = graph->createTuple(inputs);
  return handler(graph, inputs_node);
}

torch::jit::Node *
createCustomOperation(torch::jit::Graph *graph,
                      const std::vector<torch::jit::Value *> &inputs,
                      const std::string &name, const std::string &domain,
                      std::int64_t domainVersion, std::int64_t numOutputs,
                      const std::string &attributes_id_str);

torch::jit::Node *createCast(torch::jit::Graph *graph, torch::jit::Value *A,
                             c10::ScalarType scalar);

torch::jit::Node *createInternalCast(torch::jit::Graph *graph,
                                     torch::jit::Value *A,
                                     const std::string &type);

torch::jit::Node *createConstantPad(torch::jit::Graph *graph,
                                    torch::jit::Value *A,
                                    const std::vector<int64_t> &pad_shape,
                                    float constant,
                                    bool direct_pad_shape_input = false);

torch::jit::Node *createReflectionPad(torch::jit::Graph *graph,
                                      torch::jit::Value *A,
                                      const std::vector<int64_t> &pad_shape);

torch::jit::Node *createEdgePad(torch::jit::Graph *graph, torch::jit::Value *A,
                                const std::vector<int64_t> &pad_shape);

torch::jit::Node *createAddNotInPlace(torch::jit::Graph *graph,
                                      torch::jit::Value *A,
                                      torch::jit::Value *B);

torch::jit::Node *createStartForLoop(torch::jit::Graph *graph,
                                     torch::jit::Value *inputs);

torch::jit::Node *createEndForLoop(torch::jit::Graph *graph,
                                   torch::jit::Value *outputs,
                                   torch::jit::Value *inputs,
                                   std::int64_t trip_count);

torch::jit::Node *createAddUntypedInputTensor(torch::jit::Graph *graph,
                                              torch::jit::Value *input);

// Create an add output to mark a node of being an output of a subgraph.
torch::jit::Node *createAddOutputTensor(torch::jit::Graph *graph,
                                        torch::jit::Value *output);

torch::jit::Value *wrapInConstantVec(torch::jit::Graph *graph,
                                     const std::vector<int64_t> &data);

template <typename... Elms>
using FirstElmType = typename std::tuple_element<0, std::tuple<Elms...>>::type;

template <
    typename... Ints,
    std::enable_if_t<std::is_integral<FirstElmType<Ints...>>::value, int> = 0>
torch::jit::Value *wrapInConstant1D(torch::jit::Graph *graph, Ints... values) {
  std::vector<int64_t> data{std::forward<Ints>(values)...};
  return wrapInConstantVec(graph, data);
}

template <typename T> struct CreateCast {};

template <> struct CreateCast<float> {
  torch::jit::Node *operator()(torch::jit::Graph *graph,
                               torch::jit::Value *value) {
    return createCast(graph, value, c10::kFloat);
  }
};

template <> struct CreateCast<std::int32_t> {
  torch::jit::Node *operator()(torch::jit::Graph *graph,
                               torch::jit::Value *value) {
    return createCast(graph, value, c10::kInt);
  }
};

template <> struct CreateCast<std::int64_t> {
  torch::jit::Node *operator()(torch::jit::Graph *graph,
                               torch::jit::Value *value) {
    return createCast(graph, value, c10::kLong);
  }
};

template <typename T>
torch::jit::Node *castToType(torch::jit::Graph *graph,
                             torch::jit::Value *value) {
  return CreateCast<T>{}(graph, value);
}

torch::jit::Node *
createOptimizerGroup(torch::jit::Graph *graph, std::uint64_t group,
                     const std::vector<torch::jit::Value *> &list_of_params);

torch::jit::Node *createRecomputationCheckpoint(torch::jit::Graph *graph,
                                                torch::jit::Value *value);

torch::jit::Node *createUnfold(torch::jit::Graph *graph,
                               torch::jit::Value *value, int64_t dimension,
                               int64_t size, int64_t step);

torch::jit::Node *
createRandomNormal(torch::jit::Graph *graph,
                   const std::vector<torch::jit::Value *> &possible_inputs,
                   const std::vector<int64_t> &shape, float mean, float scale,
                   at::ScalarType dataType = at::ScalarType::Undefined);

torch::jit::Node *
createRandomUniform(torch::jit::Graph *graph, torch::jit::Value *possible_input,
                    const std::vector<int64_t> &shape, float high, float low,
                    at::ScalarType dataType = at::ScalarType::Undefined);

torch::jit::Node *createPrintIpuTensor(torch::jit::Graph *graph,
                                       torch::jit::Value *value,
                                       const std::string &title);

torch::jit::Node *createCallCpuOp(torch::jit::Graph *graph,
                                  const std::vector<torch::jit::Value *> &value,
                                  const std::string &id,
                                  torch::jit::Node *node);

torch::jit::Node *createSetAvailableMemory(torch::jit::Graph *graph,
                                           torch::jit::Value *value,
                                           float proportion);

torch::jit::Node *createSetMatMulSerialization(torch::jit::Graph *graph,
                                               torch::jit::Value *matmul,
                                               const std::string &mode,
                                               int64_t factor,
                                               bool keep_precision);

torch::jit::Node *createBeginIpuBlock(torch::jit::Graph *graph,
                                      std::uint64_t stage, std::int64_t phase,
                                      std::int64_t ipu);

torch::jit::Node *createMultiConvPart(torch::jit::Graph *graph,
                                      torch::jit::Node *conv_node);

torch::jit::Node *createGru(torch::jit::Graph *graph,
                            const std::vector<torch::jit::Value *> &args);

torch::jit::Node *createRnn(torch::jit::Graph *graph,
                            const std::vector<torch::jit::Value *> &args,
                            const std::vector<std::string> &activations);

// Autogenerated.
#include "CompilerOps.inc.hpp"

} // namespace poptorch

#endif // INCLUDE_POPTORCH_OP_BUILDER_HPP
