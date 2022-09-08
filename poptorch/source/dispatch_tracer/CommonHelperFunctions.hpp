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

at::Tensor copyAndCoerceType(const at::Tensor &tensor);

// If initial_op is already an outplace op return it. Otherwise get the
// corresponding outplace op and adjust the stack so the op will be handled
// properly
c10::OperatorHandle getOutplaceOpHandle(const c10::OperatorHandle &initial_op,
                                        c10::Dispatcher &dispatcher,
                                        c10::Stack &stack);

// From the schema deduce which argument if any is inplace. Only return the
// first one which is inplace. This might include an argument of an op that
// is not truly inplace, e.g. it returns the 'out' argument in the schema
// op(Tensor self, Tensor(a!) out) -> (Tensor(a!)) even when 'self' and 'out'
// are not the same tensor.
std::vector<at::Tensor> getInplaceArguments(const c10::Stack &stack,
                                            const c10::FunctionSchema &schema);

// Using the schema definition as a guide look up all the correct
// torch::jit::Values in the stack and create a jit node with the correct
// symbol. Input values from the stack are also inserted into the graph.
torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper);

// Return a string containing the tensor sizes and type.
std::string toString(const at::Tensor &t);

bool isHalfTensor(const at::Tensor &t);

at::ScalarType scalarTypeOrDefault(c10::optional<at::ScalarType> dtype);

// If device is set: return device, otherwise return the default device (ipu0)
c10::Device deviceOrDefaultIpu(c10::optional<c10::Device> device);

} // namespace poptorch

#endif // POPTORCH_DISPATCH_COMMON_HELPERS_HPP_
