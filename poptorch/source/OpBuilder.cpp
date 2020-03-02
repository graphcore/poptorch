#include <poptorch/OpBuilder.hpp>
#include <torch/csrc/jit/ir.h>

namespace poptorch {

torch::jit::Node *CreateConvolution(
    torch::jit::Graph &graph, const std::vector<torch::jit::Value *> &args,
    const std::vector<int64_t> &dilations, int64_t group,
    const std::vector<int64_t> &pads, const std::vector<int64_t> &strides) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::convolution"), args);

  newNode->is_(c10::Symbol::fromQualString("attr::dilation"), dilations);
  newNode->i_(c10::Symbol::fromQualString("attr::group"), group);
  newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"), {});
  newNode->is_(c10::Symbol::fromQualString("attr::pads"), pads);
  newNode->is_(c10::Symbol::fromQualString("attr::strides"), strides);

  return newNode;
}

torch::jit::Node *CreateBatchNorm(torch::jit::Graph &graph,
                                  const std::vector<torch::jit::Value *> &args,
                                  float epsilon, float momentum,
                                  bool training) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::batchnorm"), args);

  newNode->i_(c10::Symbol::fromQualString("attr::num_outputs"), 1);
  newNode->f_(c10::Symbol::fromQualString("attr::epsilon"), epsilon);
  newNode->f_(c10::Symbol::fromQualString("attr::momentum"), momentum);
  newNode->i_(c10::Symbol::fromQualString("attr::training"), training);
  return newNode;
}

torch::jit::Node *CreateMaxPool(torch::jit::Graph &graph,
                                torch::jit::Value *input,
                                const std::vector<int64_t> &kernel_size,
                                const std::vector<int64_t> &strides,
                                const std::vector<int64_t> &padding,
                                const std::vector<int64_t> &dilations) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::max_pool"), {input});

  newNode->i_(c10::Symbol::fromQualString("attr::num_outputs"), 1);
  newNode->is_(c10::Symbol::fromQualString("attr::kernel_size"), kernel_size);
  newNode->is_(c10::Symbol::fromQualString("attr::padding"), padding);
  newNode->i_(c10::Symbol::fromQualString("attr::storage_order"), 0);
  newNode->is_(c10::Symbol::fromQualString("attr::strides"), strides);
  return newNode;
}

torch::jit::Node *CreateAdd(torch::jit::Graph &graph, torch::jit::Value *A,
                            torch::jit::Value *B) {
  return graph.create(c10::Symbol::fromQualString("popart::add"), {A, B});
}

torch::jit::Node *CreateMatmul(torch::jit::Graph &graph, torch::jit::Value *A,
                               torch::jit::Value *B) {
  return graph.create(c10::Symbol::fromQualString("popart::matmul"), {A, B});
}

torch::jit::Node *CreateFlatten(torch::jit::Graph &graph,
                                torch::jit::Value *A) {

  return graph.create(c10::Symbol::fromQualString("popart::flatten"), {A});
}

torch::jit::Node *CreateGEMM(torch::jit::Graph &graph, torch::jit::Value *A,
                             torch::jit::Value *B, torch::jit::Value *C,
                             float beta, float alpha) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::gemm"), {A, B, C});

  newNode->f_(c10::Symbol::fromQualString("attr::beta"), beta);
  newNode->f_(c10::Symbol::fromQualString("attr::alpha"), alpha);
  newNode->i_(c10::Symbol::fromQualString("attr::transA"), 0);
  newNode->i_(c10::Symbol::fromQualString("attr::transB"), 0);

  return newNode;
}

torch::jit::Node *CreateAveragePool(torch::jit::Graph &graph,
                                    torch::jit::Value *A,
                                    const std::vector<int64_t> &kernel_shape,
                                    const std::vector<int64_t> &stride,
                                    const std::vector<int64_t> &padding) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::average_pool"), {A});

  newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"), kernel_shape);
  newNode->i_(c10::Symbol::fromQualString("attr::count_include_pad"), 0);
  newNode->is_(c10::Symbol::fromQualString("attr::pads"), padding);
  newNode->is_(c10::Symbol::fromQualString("attr::strides"), stride);

  return newNode;
}


torch::jit::Node *CreateSoftmax(torch::jit::Graph &graph,
                                  torch::jit::Value *A, std::int64_t dim){
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::softmax"), {A});
  newNode->i_(c10::Symbol::fromQualString("attr::dim"), dim);
  return newNode;
}


torch::jit::Node *CreateReshape(torch::jit::Graph &graph,
                                  torch::jit::Value *A,  const std::vector<int64_t>& new_shape) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::reshape"), {A});
  newNode->is_(c10::Symbol::fromQualString("attr::shape"), new_shape);
  return newNode;
}

torch::jit::Node *CreateDropout(torch::jit::Graph &graph, torch::jit::Value *A, float rate) {
  torch::jit::Node *newNode =
      graph.create(c10::Symbol::fromQualString("popart::dropout"), {A});
  newNode->i_(c10::Symbol::fromQualString("attr::num_outputs"), 1);
  newNode->f_(c10::Symbol::fromQualString("attr::rate"), rate);
  return newNode;
}


} // namespace poptorch