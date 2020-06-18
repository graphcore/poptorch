// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart_compiler/Compiler.hpp>

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <popart/builder.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensors.hpp>
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

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

  // Record each loss as it is used so we can make them inputs of the global
  // identity op.
  std::vector<popart::TensorId> losses;

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

  popart::TensorId intConstant(const std::vector<popart::TensorId> &inputs,
                               const std::vector<int64_t> &data,
                               const std::vector<int64_t> &shape);

  popart::TensorId floatConstant(const std::vector<popart::TensorId> &inputs,
                                 const std::vector<double> &data,
                                 const std::vector<int64_t> &shape);
};

popart::TensorId
CompilerImpl::reshape(const std::vector<popart::TensorId> &inputs,
                      const std::vector<int64_t> &shape) {
  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return opBuilder->reshape_const(aiOnnx, inputs, shape);
}

popart::TensorId
CompilerImpl::intConstant(const std::vector<popart::TensorId> &inputs,
                          const std::vector<int64_t> &data,
                          const std::vector<int64_t> &shape) {
  UNUSED(inputs);
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{"INT32", shape};

  std::int64_t totalSize = std::accumulate(shape.begin(), shape.end(), 1,
                                           std::multiplies<std::int64_t>());
  std::vector<int64_t> broadcastedData(totalSize);

  // Create the inital data for the variable.
  popart::ConstVoidData theData;

  if (data.size() == 1 && totalSize != 1) {
    std::for_each(broadcastedData.begin(), broadcastedData.end(),
                  [&data](std::int64_t &i) { i = data[0]; });

    theData.data = broadcastedData.data();
    theData.info = info;
  } else {
    theData.data = data.data();
    theData.info = info;
  }

  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return aiOnnx.constant(theData);
}

popart::TensorId
CompilerImpl::floatConstant(const std::vector<popart::TensorId> &inputs,
                            const std::vector<double> &data,
                            const std::vector<int64_t> &shape) {
  UNUSED(inputs);
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{"FLOAT", shape};

  std::int64_t totalSize = std::accumulate(shape.begin(), shape.end(), 1,
                                           std::multiplies<std::int64_t>());
  std::vector<float> broadcastedData(totalSize);

  // Create the inital data for the variable.
  popart::ConstVoidData theData;

  if (data.size() == 1 && totalSize != 1) {
    std::for_each(broadcastedData.begin(), broadcastedData.end(),
                  [&data](float &i) { i = data[0]; });

    theData.data = broadcastedData.data();
    theData.info = info;
  } else {
    int counter = 0;
    std::for_each(broadcastedData.begin(), broadcastedData.end(),
                  [&](float &i) { i = data[counter++]; });

    theData.data = broadcastedData.data();
    theData.info = info;
  }

  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return aiOnnx.constant(theData);
}

} // namespace detail

// Variadic output case. For now we will add all outputs to the graph and
// allocate them on the same IPU but we will only return one. This means only
// one output can be used by user IR (but can still be used by the backed via
// transformations).
template <typename T> struct HandleOutput {
  poptorch::TensorId operator()(T &in, bool loss, detail::CompilerImpl *impl) {
    std::set<popart::TensorId> ids;

    for (popart::TensorId id : in) {
      ids.insert(id);
      impl->ids.push_back(id);

      if (loss) {
        impl->losses.push_back(id);
      }
    }

    impl->opBuilder->virtualGraph(ids, impl->activeIpu);
    impl->usedIpus.insert(impl->activeIpu);

    // Return the first added tensor as the sole return of this IR op.
    return impl->ids.size() - in.size();
  }
};

// Single tensor output case
template <> struct HandleOutput<popart::TensorId> {
  poptorch::TensorId operator()(popart::TensorId in, bool loss,
                                detail::CompilerImpl *impl) {
    impl->opBuilder->virtualGraph(in, impl->activeIpu);
    impl->usedIpus.insert(impl->activeIpu);
    impl->ids.push_back(in);

    if (loss) {
      impl->losses.push_back(in);
    }

    return impl->ids.size() - 1;
  }
};

poptorch::TensorId
Compiler::AddInputTensor(const char *string,
                         const std::vector<std::int64_t> &dims) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{string, dims};
  impl->ids.push_back(impl->opBuilder->addInputTensor(info));
  return impl->ids.size() - 1;
}

// A whitelist of supported loss operations. Popart needs to know which
// operations are losses so they can be marked by the session.
static bool IsLoss(const std::string &operation) {
  if (operation == "popart::l1loss" || operation == "popart::nllloss" ||
      operation == "popart::identityloss") {
    return true;
  }

  return false;
}

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<double>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define NONE
#define ARG(Type, Name) , Type Name
#define BODY_ARG(Name) , Name
// Create a function decl with the given call and arguments.
#define OP_DECL(ns, funcName, function, onnxImpl, Args, BodyArgs)              \
  poptorch::TensorId Compiler::function(                                       \
      const std::vector<poptorch::TensorId> &inputs Args) {                    \
    auto AiOnnxOpset9 = impl->opBuilder->aiOnnxOpset9();                       \
    auto AiGraphcoreOpset1 = impl->opBuilder->aiGraphcoreOpset1();             \
    const bool isLoss = IsLoss(#ns "::" #funcName);                            \
    std::vector<popart::TensorId> ins;                                         \
    std::transform(                                                            \
        inputs.begin(), inputs.end(), std::back_inserter(ins),                 \
        [&](poptorch::TensorId index) { return impl->ids[index]; });           \
    auto output = onnxImpl(ins BodyArgs);                                      \
    return HandleOutput<decltype(output)>{}(output, isLoss, impl.get());       \
  }

#include "popart_compiler/SupportedOperations.inc.h"
#undef BODY_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef INT_VEC
#undef FLOAT_VEC
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

  popart::MutableVoidData mutableData;
  mutableData.data = data;
  mutableData.info = info;

  impl->weightCallback.insert(id, mutableData);

  return impl->ids.size() - 1;
}

void Compiler::AddOutput(poptorch::TensorId output) {
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

void Compiler::SetUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int32_t>>(
          static_cast<std::int32_t *>(ptr), dims));

  impl->popartOutgoing.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::InitSession(bool profile) {
  // Try and get a single IPU. If not avaliable, run on CPU.
  // TODO: Make an actual device selection mechanism.
  std::shared_ptr<popart::DeviceInfo> device =
      popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
          impl->usedIpus.size());

  if (!device) {
    logging::debug(
        "No IPU device found, falling back to CPU emulator (IPU Model)");
    device = popart::DeviceManager::createDeviceManager().createCpuDevice();
  } else {
    logging::debug("Acquired IPU device, running on device.");
  }

  popart::SessionOptions options;

  options.logDir = ".";

  if (impl->usedIpus.size() > 1) {
    options.enablePipelining = true;
    options.virtualGraphMode = popart::VirtualGraphMode::Manual;
  }

  if (impl->gradientAccumulation > 1) {
    options.enableGradientAccumulation = true;
    options.accumulationFactor = impl->gradientAccumulation;
  }

  // Create the anchors, these are used to copy to the host.
  auto dataFlow = popart::DataFlow(impl->steps, impl->anchors);

  // Create the popart session object to actually run the graph.
  if (!impl->isTraining) {
    options.constantWeights = false;

    // Create an inference session.
    impl->session = popart::InferenceSession::createFromOnnxModel(
        impl->opBuilder->getModelProto(), dataFlow, device, {}, options,
        popart::PatternsLevel::Default);
  } else {
    auto optimizer = popart::ConstSGD(0.001);

    // Set a global identity loss that all other losses derive from.
    popart::TensorId lossRoot =
        impl->opBuilder->aiGraphcoreOpset1().identityloss(impl->losses);
    impl->opBuilder->virtualGraph(lossRoot, impl->activeIpu);

    popart::GraphTransformer transformer{impl->opBuilder->getModelProto()};

    transformer.prepareNodesForTraining();

    // Create the training session.
    impl->session = popart::TrainingSession::createFromOnnxModel(
        transformer.getModelProto(), dataFlow, lossRoot, optimizer, device, {},
        options, popart::PatternsLevel::Default);
  }

  logging::trace(
      "Popart serialised IR:\n{}",
      impl->session->serializeIr(popart::IrSerializationFormat::JSON));

  // Poplar compilation.
  try {
    logging::trace("Begining Poplar compilation.");
    impl->session->prepareDevice();
    logging::trace("Finished Poplar compilation.");
  } catch (popart::memory_allocation_err &e) {
    std::ofstream stream;
    stream.open("OOMReport.json");
    stream << e.getGraphReport(true);
    stream.close();

    std::rethrow_exception(std::current_exception());
  }

  if (profile) {
    std::ofstream stream;
    stream.open("GraphReport.json");
    stream << impl->session->getGraphReport();
    stream.close();
  }

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

std::uint64_t Compiler::PopartBatchDim() const {
  return impl->replicationFactor * impl->steps * impl->gradientAccumulation;
}

Compiler::Compiler(Compiler &&other) { impl = std::move(other.impl); }

Compiler::Compiler(bool isTraining, std::uint64_t steps,
                   std::uint64_t replicationFactor,
                   std::uint64_t gradientAccumulation) {
  impl = std::make_unique<detail::CompilerImpl>();
  impl->isTraining = isTraining;
  impl->steps = steps;
  impl->replicationFactor = replicationFactor;
  impl->gradientAccumulation = gradientAccumulation;
}

Compiler::~Compiler() {}

} // namespace poptorch
