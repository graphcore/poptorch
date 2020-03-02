#ifndef INCLUDE_POPTORCH_OP_BUILDER_HPP
#define INCLUDE_POPTORCH_OP_BUILDER_HPP

#include <string>
#include <vector>

namespace torch {
namespace jit {
class Graph;
class Node;
class Value;
} // namespace jit
} // namespace torch

namespace poptorch {

torch::jit::Node *CreateConvolution(
    torch::jit::Graph &graph, const std::vector<torch::jit::Value *> &args,
    const std::vector<int64_t> &dilations, int64_t group,
    const std::vector<int64_t> &pads, const std::vector<int64_t> &strides);

torch::jit::Node *CreateBatchNorm(torch::jit::Graph &graph,
                                  const std::vector<torch::jit::Value *> &args,
                                  float epsilon, float momentum, bool training);

torch::jit::Node *CreateMaxPool(torch::jit::Graph &graph,
                                torch::jit::Value *input,
                                const std::vector<int64_t> &kernel_size,
                                const std::vector<int64_t> &strides,
                                const std::vector<int64_t> &padding,
                                const std::vector<int64_t> &dilations);

torch::jit::Node *CreateAdd(torch::jit::Graph &graph, torch::jit::Value *A,
                            torch::jit::Value *B);

torch::jit::Node *CreateMatmul(torch::jit::Graph &graph, torch::jit::Value *A,
                               torch::jit::Value *B);

torch::jit::Node *CreateFlatten(torch::jit::Graph &graph, torch::jit::Value *A);

torch::jit::Node *CreateGEMM(torch::jit::Graph &graph, torch::jit::Value *A,
                             torch::jit::Value *B, torch::jit::Value *C,
                             float beta, float alpha);

torch::jit::Node *CreateAveragePool(torch::jit::Graph &graph,
                                    torch::jit::Value *A,
                                    const std::vector<int64_t> &kernel_shape,
                                    const std::vector<int64_t> &stride,
                                    const std::vector<int64_t> &padding);

torch::jit::Node *CreateSoftmax(torch::jit::Graph &graph, torch::jit::Value *A,
                                std::int64_t dim);

torch::jit::Node *CreateReshape(torch::jit::Graph &graph, torch::jit::Value *A,
                                const std::vector<int64_t> &new_shape);


torch::jit::Node *CreateReshape(torch::jit::Graph &graph, torch::jit::Value *A,
                                const std::vector<int64_t> &new_shape);

torch::jit::Node *CreateDropout(torch::jit::Graph &graph, torch::jit::Value *A, float rate);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_OP_BUILDER_HPP
