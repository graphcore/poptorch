// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_COMMON_HELPERS_HPP_
#define POPTORCH_DISPATCH_COMMON_HELPERS_HPP_

#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
struct Graph;
struct Node;
struct Value;
} // namespace jit
} // namespace torch

namespace poptorch {

class ValueMapper;

// Insert a poptorch::tensor_constant into the graph which doesn't require
// further canonicalization.
torch::jit::Value *makeConstant(torch::jit::Graph &graph,
                                const at::Tensor &tensor);

at::Tensor copyAndCoerceType(const at::Tensor &tensor);

c10::OperatorHandle getOutplaceOpHandle(const c10::OperatorHandle &initial_op,
                                        c10::Dispatcher &dispatcher);

// From the schema deduce which argument if any is inplace. Only return the
// first one which is inplace. This might include an argument of an op that
// is not truly inplace, e.g. it returns the 'out' argument in the schema
// op(Tensor self, Tensor(a!) out) -> (Tensor(a!)) even when 'self' and 'out'
// are not the same tensor.
c10::intrusive_ptr<at::TensorImpl>
getInplaceArgument(const c10::Stack &stack, const c10::FunctionSchema &schema);

// Using the schema definition as a guide look up all the correct
// torch::jit::Values in the stack and create a jit node with the correct
// symbol. Input values from the stack are also inserted into the graph.
torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper);

void fixNodeOutput(torch::jit::Node *node, const c10::Stack &stack,
                   ValueMapper &mapper);

// Run our canonicaliser passes for the aten_target over the graph.
torch::jit::Node *canonicalise(const c10::FunctionSchema &schema,
                               torch::jit::Node *aten_target,
                               torch::jit::Graph &graph,
                               bool is_allowed_to_fail);

// If this value was replaced with another by the most recently run handler,
// return the replacement. If not, return nullptr.
torch::jit::Value *wasReplaced(torch::jit::Value *target);

// Return a string containing the tensor sizes and type.
std::string toString(const at::Tensor &t);

bool isHalfTensor(const at::Tensor &t);

at::ScalarType scalarTypeOrDefault(c10::optional<at::ScalarType> dtype);

} // namespace poptorch

#endif // POPTORCH_DISPATCH_COMMON_HELPERS_HPP_
