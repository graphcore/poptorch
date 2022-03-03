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

bool getOutplaceOpHandle(c10::OperatorHandle &initial_op,
                         c10::Dispatcher &dispatcher);

// From the schema deduce which argument if any is inplace. Only return the
// first one which is inplace. This might include an argument of an op that
// is not truly inplace, e.g. it returns the 'out' argument in the schema
// op(Tensor self, Tensor(a!) out) -> (Tensor(a!)) even when 'self' and 'out'
// are not the same tensor.
c10::intrusive_ptr<at::TensorImpl>
getInplaceArgument(const c10::Stack &stack, const c10::FunctionSchema &schema);

// Unlike 'getInplaceArgument', returns true if and only if the
// op is truly inplace.
bool isTrulyInplace(const c10::Stack &stack, const c10::FunctionSchema &schema);

// Using the schema definition as a guide look up all the correct
// torch::jit::Values in the stack and create a jit node with the correct
// symbol. Input values from the stack are also inserted into the graph.
torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper);

bool shouldRunOnCpu(bool is_inplace, const std::string &op_name);

class HalfFloatConverter {
public:
  explicit HalfFloatConverter(c10::Stack *stack,
                              const c10::FunctionSchema &schema,
                              ValueMapper &mapper);

  void pre();
  void post();

private:
  c10::Stack *_stack;
  const c10::FunctionSchema &_schema;
  ValueMapper &_mapper;
  std::unordered_map<c10::intrusive_ptr<at::TensorImpl>,
                     c10::intrusive_ptr<at::TensorImpl>>
      _half_map;
  bool _has_half_input;
};

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

} // namespace poptorch

#endif // POPTORCH_DISPATCH_COMMON_HELPERS_HPP_
