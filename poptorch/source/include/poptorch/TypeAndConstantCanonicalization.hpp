// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_TYPE_AND_CONSTANT_CANONICALIZATION_H
#define INCLUDE_POPTORCH_TYPE_AND_CONSTANT_CANONICALIZATION_H

#include <sstream>
#include <string>
#include <vector>

namespace at {
class Tensor;
} // namespace at

namespace c10 {
struct Symbol;
} // namespace c10

namespace torch {
namespace jit {
struct Graph;
struct Node;
} // namespace jit
} // namespace torch

namespace poptorch {
namespace type_and_constant_canonicalization {

// Add the number of elements of the list to the type by replacing it with
// ListTypeWithNumElements instances. The PyTorch ListType does not contain
// the number of elements. If revert is "true", reverts all such types to the
// original ListType.
void addListNumElements(torch::jit::Graph *graph, bool revert = false);

void evaluateConstexprs(torch::jit::Graph *graph);

// Turn non-floating point parameters into constants as these are not supported
// in popart. The pass also removes the affected graph inputs and modifies
// 'parameter_names' and 'traced_parameter_tensors' accordingly.
void makeConstantIntParams(torch::jit::Graph *graph,
                           std::vector<std::string> &parameter_names,
                           std::vector<at::Tensor> &traced_parameter_tensors);

// Change the graph to add a poptorch::host_side_cast node after every graph
// input whose type is unsupported (Long, Double, BFloat16) to reflect the
// casting which would happen on the host and the correct types as they
// would be on the graph.
void castUnsupportedInputs(torch::jit::Graph *graph);

// Change any unsupported output types to the appropriate equivalent (e.g.
// double to float) and warn; error on any totally unsupported types e.g. 8 bit.
void checkAndChangeOutputTypes(torch::jit::Graph *graph);

// Changes all constants used in implicit casting operations into tensor
// constants (poptorch::tensor_constant) of the correct type.
void canonicaliseConstants(torch::jit::Graph *graph,
                           std::vector<std::size_t> &input_index_map);

} // namespace type_and_constant_canonicalization
} // namespace poptorch

#endif // INCLUDE_POPTORCH_TYPE_AND_CONSTANT_CANONICALIZATION_H
