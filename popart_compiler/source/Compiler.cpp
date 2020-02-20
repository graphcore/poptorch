#include <popart_compiler/Compiler.hpp>

#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/matmul.hpp>
#include <popart/session.hpp>
#include <popart/tensors.hpp>

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace poptorch {

namespace detail {

struct CompilerImpl {
public:
  friend Compiler;

  CompilerImpl() : opBuilder(popart::Builder::create()) {}

  std::unique_ptr<popart::Builder> opBuilder;

  std::map<popart::TensorId, popart::AnchorReturnType> anchors;

  std::vector<popart::TensorId> ids;

  // Input tensors to the session.
  std::map<popart::TensorId, popart::IArray &> popartIncoming;

  // Output tensors for the session.
  std::map<popart::TensorId, popart::IArray &> popartOutgoing;

  std::unique_ptr<popart::Session> session;
};

} // namespace detail

poptorch::TensorId
Compiler::AddInputTensor(const char *string,
                         const std::vector<std::int64_t> &dims) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{string, dims};
  impl->ids.push_back(impl->opBuilder->addInputTensor(info));
  return impl->ids.size() - 1;
}

poptorch::TensorId
Compiler::BuildOp(const char *operation,
                  const std::vector<poptorch::TensorId> &inputs) {
  std::string op{operation};

  // Convert from the impl id class to the popart id, this is just a indexing op
  // as the impl id is the index of the popart id.
  std::vector<popart::TensorId> toPopartIds;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(toPopartIds),
                 [&](poptorch::TensorId index) { return impl->ids[index]; });

  auto aiOnnx = impl->opBuilder->aiOnnxOpset9();

  // Add the operation to the graph and to the list of popart ids.
  if (op == "aten::t") {
    impl->ids.push_back(aiOnnx.transpose(toPopartIds));
  } else if (op == "aten::matmul") {
    impl->ids.push_back(aiOnnx.matmul(toPopartIds, "MatMul"));
  } else if (op == "aten::add") {
    impl->ids.push_back(aiOnnx.add({toPopartIds[0], toPopartIds[1]}, "Add"));
  } else if (op == "aten::relu") {
    impl->ids.push_back(aiOnnx.relu(toPopartIds, "Relu"));
  } else if (op == "prim::Constant") {
    // Ignore these constants I think we should eliminate them in earlier
    // passes.
    impl->ids.push_back("");
  }

  // Return the index of the operation.
  return impl->ids.size() - 1;
}

poptorch::TensorId
Compiler::AddInitializedInputTensor(const char *name, const char *type,
                                    const std::vector<std::int64_t> &dims,
                                    void *data) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{type, dims};

  // Create the inital data for the variable.
  popart::ConstVoidData theData;
  theData.data = data;
  theData.info = info;

  impl->ids.push_back(
      impl->opBuilder->addInitializedInputTensor(theData, name));

  return impl->ids.size() - 1;
}

void Compiler::AddOutput(poptorch::TensorId output) {
  impl->opBuilder->addOutputTensor(impl->ids[output]);

  impl->anchors.insert({impl->ids[output], popart::AnchorReturnType("FINAL")});
}

void Compiler::SetUpInputOp(poptorch::TensorId id, void *ptr,
                            const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  // TODO: Obviously get rid of this heap alloc.
  popart::NDArrayWrapper<float> *data =
      new popart::NDArrayWrapper<float>{static_cast<float *>(ptr), dims};
  impl->popartIncoming.insert({impl->ids[id], *data});
}

void Compiler::SetUpOutputOp(poptorch::TensorId id, void *ptr,
                             const std::vector<std::int64_t> &dims) {

  // Popart wrapper around the tensor pointer.
  // TODO: Obviously get rid of this heap alloc.
  popart::NDArrayWrapper<float> *data =
      new popart::NDArrayWrapper<float>{static_cast<float *>(ptr), dims};
  impl->popartOutgoing.insert({impl->ids[id], *data});
}

void Compiler::InitSession() {

  // Create the anchors, these are used to copy to the host.
  auto dataFlow = popart::DataFlow(1, impl->anchors);

  // Create a CPU device for now.
  // TODO: Make an actual device selection mechanism.
  std::shared_ptr<popart::DeviceInfo> cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  // Create the popart session object to actually run the graph.
  impl->session = popart::InferenceSession::createFromOnnxModel(
      impl->opBuilder->getModelProto(), dataFlow, cpuDevice, {}, {}, {},
      popart::PatternsLevel::NONE);

  impl->session->prepareDevice();
}

void Compiler::Run() {
  // Execute the model on IPU.
  popart::StepIO stepio(impl->popartIncoming, impl->popartOutgoing);
  impl->session->run(stepio);
}

std::vector<std::int64_t> Compiler::GetSize(poptorch::TensorId id) {
  popart::TensorInfo info = impl->session->getInfo(impl->ids[id]);

  return info.shape();
}

Compiler::Compiler() { impl = std::make_unique<detail::CompilerImpl>(); }


Compiler::~Compiler() {}

} // namespace poptorch
