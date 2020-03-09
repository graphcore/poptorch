#include <popart_compiler/Compiler.hpp>

#include <fstream>

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <popart/builder.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensors.hpp>
#include <unordered_map>
#include <vector>

namespace poptorch {

namespace detail {

struct CompilerImpl {
public:
  friend Compiler;

  CompilerImpl() : opBuilder(popart::Builder::create()), activeIpu(0) {}

  std::unique_ptr<popart::Builder> opBuilder;

  std::map<popart::TensorId, popart::AnchorReturnType> anchors;

  std::vector<popart::TensorId> ids;

  // Input tensors to the session.
  std::map<popart::TensorId, popart::IArray &> popartIncoming;

  // Output tensors for the session.
  std::map<popart::TensorId, popart::IArray &> popartOutgoing;

  std::list<popart::TensorId> outputs;

  // A list to allocate our buffers in so they get released.
  std::list<std::unique_ptr<popart::IArray>> memoryManager;

  std::unique_ptr<popart::Session> session;

  popart::WeightsIO weightCallback;

  bool isTraining;
  std::uint64_t steps;
  std::uint64_t replicationFactor;
  std::uint64_t gradientAccumulation;

  // We add operations using a state based system so the user would set the
  // active IPU and all subsequent operations will be added to that IPU until
  // stopped.
  std::uint64_t activeIpu;

  std::unordered_set<std::uint64_t> usedIpus;

  // Domain helpers
  popart::TensorId reshape(const std::vector<popart::TensorId> &inputs,
                           const std::vector<int64_t> &shape);
};

popart::TensorId
CompilerImpl::reshape(const std::vector<popart::TensorId> &inputs,
                      const std::vector<int64_t> &shape) {
  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return opBuilder->reshape_const(aiOnnx, inputs, shape);
}

} // namespace detail

poptorch::TensorId
Compiler::AddInputTensor(const char *string,
                         const std::vector<std::int64_t> &dims) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{string, dims};
  impl->ids.push_back(impl->opBuilder->addInputTensor(info));
  return impl->ids.size() - 1;
}

#define INT_VEC std::vector<std::int64_t>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define NONE
#define ARG(Type, Name) , Type Name
#define BODY_ARG(Name) , Name
// Create a function decl with the given call and arguments.
#define OP_DECL(name, function, onnxImpl, Args, BodyArgs, VariadicIndex)       \
  poptorch::TensorId Compiler::function(                                       \
      const std::vector<poptorch::TensorId> &inputs Args) {                    \
    auto aiOnnx = impl->opBuilder->aiOnnxOpset9();                             \
    auto aiGraphcore = impl->opBuilder->aiGraphcoreOpset1();                   \
    std::vector<popart::TensorId> ins;                                         \
    std::transform(                                                            \
        inputs.begin(), inputs.end(), std::back_inserter(ins),                 \
        [&](poptorch::TensorId index) { return impl->ids[index]; });           \
    impl->ids.push_back(onnxImpl(ins BodyArgs) VariadicIndex);                 \
    const poptorch::TensorId id = impl->ids.size() - 1;                        \
    impl->opBuilder->virtualGraph(impl->ids[id], impl->activeIpu);             \
    impl->usedIpus.insert(impl->activeIpu);                                    \
    return id;                                                                 \
  }

#include "popart_compiler/SupportedOperations.inc.h"
#undef BODY_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef INT_VEC
#undef FLOAT
#undef INT
#undef BOOL

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

  popart::TensorId id = impl->ids[impl->ids.size() - 1];

  // std::cout << "Tensor ID: " << id << " has address: " << data << std::endl;

  popart::MutableVoidData mutableData;
  mutableData.data = data;
  mutableData.info = info;

  impl->weightCallback.insert(id, mutableData);

  return impl->ids.size() - 1;
}

void Compiler::AddOutput(poptorch::TensorId output) {
  impl->opBuilder->addOutputTensor(impl->ids[output]);

  impl->outputs.push_back(impl->ids[output]);

  impl->anchors.insert({impl->ids[output], popart::AnchorReturnType("ALL")});
}

void Compiler::SetUpInputOp(poptorch::TensorId id, float *ptr,
                            const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(std::make_unique<popart::NDArrayWrapper<float>>(
      static_cast<float *>(ptr), dims));
  impl->popartIncoming.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::SetUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                            const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int32_t>>(ptr, dims));
  impl->popartIncoming.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::SetUpInputOp(poptorch::TensorId id, std::int64_t *ptr,
                            const std::vector<std::int64_t> &dims) {

  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int64_t>>(ptr, dims));
  impl->popartIncoming.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::SetUpOutputOp(poptorch::TensorId id, float *ptr,
                             const std::vector<std::int64_t> &dims) {

  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(std::make_unique<popart::NDArrayWrapper<float>>(
      static_cast<float *>(ptr), dims));

  impl->popartOutgoing.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::InitSession() {
  // Try and get a single IPU. If not avaliable, run on CPU.
  // TODO: Make an actual device selection mechanism.
  std::shared_ptr<popart::DeviceInfo> device =
      popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
          impl->usedIpus.size());

  if (!device) {
    std::cout << "No IPU device found, falling back to CPU emulator (IPU Model)"
              << std::endl;
    device = popart::DeviceManager::createDeviceManager().createCpuDevice();
  } else {
    std::cout << "Acquired IPU device, running on device." << std::endl;
  }


  popart::SessionOptions options;

    if (impl->usedIpus.size() > 1) {
      options.enablePipelining = true;
      options.enableVirtualGraphs = true;
      options.virtualGraphMode =  popart::VirtualGraphMode::Manual;
    }


   if (impl->gradientAccumulation > 1) {
      options.enableGradientAccumulation = true;
      options.accumulationFactor= impl->gradientAccumulation;
   }

  // Create the anchors, these are used to copy to the host.
  auto dataFlow = popart::DataFlow(impl->steps, impl->anchors);


  // Create the popart session object to actually run the graph.
  if (!impl->isTraining) {
    options.constantWeights = false;

    // Create an inference session.
    impl->session = popart::InferenceSession::createFromOnnxModel(
        impl->opBuilder->getModelProto(), dataFlow, device, {}, {}, options,
        popart::PatternsLevel::DEFAULT);
  } else {
    auto optimizer = popart::ConstSGD(0.01);

    // TODO: Some other mechanism of working out what the training label is and
    // what the output it.
    popart::TensorId networkOutput = *impl->outputs.begin();
    auto inLabels = impl->ids[1];

    // TODO: Plug the leak.
    popart::Loss *loss = new popart::NllLoss(networkOutput, inLabels, "loss",
                                             popart::ReductionType::SUM);

loss->virtualGraph(impl->activeIpu);
    popart::GraphTransformer transformer{impl->opBuilder->getModelProto()};

    transformer.prepareNodesForTraining();

    // Create the training session.
    impl->session = popart::TrainingSession::createFromOnnxModel(
        transformer.getModelProto(), dataFlow, {loss}, optimizer, device, {},
       options, popart::PatternsLevel::DEFAULT);
  }


  // Poplar compilation.
  impl->session->prepareDevice();

  impl->session->weightsFromHost();
}

void Compiler::Run() {

  // TODO don't do this everytime.
  if (!impl->isTraining) {
    impl->session->weightsFromHost();
    impl->session->writeWeights(impl->weightCallback);
  }

  // Execute the model on IPU.
  popart::StepIO stepio(impl->popartIncoming, impl->popartOutgoing);
  impl->session->run(stepio);

  // TODO don't do this everytime.
  if (impl->isTraining) {
    impl->session->weightsToHost();
    impl->session->readWeights(impl->weightCallback);
  }

  // The buffers handle the communication between pytorch and popart, we set
  // them up each run.
  // TODO: This might be annoying for performance.
  impl->popartIncoming.clear();
  impl->popartOutgoing.clear();
  impl->memoryManager.clear();
}

std::vector<std::int64_t> Compiler::GetSize(poptorch::TensorId id) {
  popart::TensorInfo info = impl->session->getInfo(impl->ids[id]);

  return info.shape();
}

void Compiler::SetActiveIpu(std::uint64_t id) { impl->activeIpu = id; }

std::uint64_t Compiler::BatchPerStep() const { return impl->steps; }


std::uint64_t Compiler::PopartBatchDim() const { return impl->replicationFactor * impl->steps * impl->gradientAccumulation; }


Compiler::Compiler(Compiler &&other) { impl = std::move(other.impl); }

Compiler::Compiler(bool isTraining, std::uint64_t steps, std::uint64_t replicationFactor,  std::uint64_t gradientAccumulation) {
  impl = std::make_unique<detail::CompilerImpl>();
  impl->isTraining = isTraining;
  impl->steps = steps;
  impl->replicationFactor = replicationFactor;
  impl->gradientAccumulation = gradientAccumulation;
}

Compiler::~Compiler() {}

} // namespace poptorch
