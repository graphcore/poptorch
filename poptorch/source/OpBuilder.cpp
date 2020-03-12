#include <poptorch/OpBuilder.hpp>
#include <torch/csrc/jit/ir.h>

namespace poptorch {

/*
 * Manually added operation.
 */
torch::jit::Node *CreateReshape(torch::jit::Graph &graph, torch::jit::Value *A,
                                const std::vector<int64_t> &new_shape) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::reshape"), {A});
  newNode->is_(c10::Symbol::fromQualString("attr::shape"), new_shape);
  return newNode;
}

/*
 * Auto generated operation.
 */
#include "CompilerOps.cpp.inc"

} // namespace poptorch
