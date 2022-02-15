// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_COMMON_HELPERS_HPP_
#define POPTORCH_DISPATCH_COMMON_HELPERS_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <string>
#include <vector>

namespace poptorch {

class ValueMapper;

// Create a node based on the Schema which deduces the return and input types
// from the inputs/stack and the name from the schema. As far as our
// canonicalisation is concerned this *is* the "aten" node it purports to be
// however it may not match it exacty, and is not created by the normal JIT
// process.
torch::jit::Node *
createAtenTarget(torch::jit::Graph &graph, const c10::FunctionSchema &schema,
                 const std::vector<torch::jit::Value *> &inputs,
                 c10::Stack *stack, ValueMapper &mapper);

// Run our canonicaliser passes over the graph.
torch::jit::Node *canonicalise(const c10::FunctionSchema &schema,
                               torch::jit::Node *aten_target,
                               torch::jit::Graph &graph,
                               bool is_allowed_to_fail);

// Using the schema definition as a guide look up all the correct
// torch::jit::Values in the stack.
torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper);

// From the schema deduce which argument if any is inplace. Only return the
// first one which is inplace.
c10::intrusive_ptr<at::TensorImpl>
getInplaceArgument(c10::Stack &stack, const c10::FunctionSchema &schema);

// Return a string containing the tensor sizes and type.
std::string toString(const at::Tensor &t);

// If this value was replaced with another by the most recently run handler,
// return the replacement. If not, return nullptr.
torch::jit::Value *wasReplaced(torch::jit::Value *target);

} // namespace poptorch

#endif // POPTORCH_DISPATCH_COMMON_HELPERS_HPP_
