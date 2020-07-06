// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_H
#define POPART_COMPILER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/PopartEnums.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

using TensorId = std::size_t;

namespace detail {
struct CompilerImpl;
struct SessionOptionsImpl;
} // namespace detail

struct OutputType {
  enum class Type { Tensor, Tuple, List };
  Type type;
  int64_t numElements{0};
};

// Extract the value from the map or return zero.
static std::pair<float, bool> FindInMapOrZero(
    const std::unordered_map<std::string, std::pair<float, bool>> &opts,
    const std::string &name) {

  // Lookup map.
  auto itr = opts.find(name);
  if (itr != opts.end()) {
    return itr->second;
  }

  logging::info("Optimizer map didn't have field for {}, defaulting to zero {}",
                name);
  return {0.0f, false};
}

struct Optimizer {
  Optimizer(
      const std::unordered_map<std::string, std::pair<float, bool>> &opts) {
    // It is valid to not pass in a optimizer.
    if (opts.empty()) {
      type = OptimizerType::NONE;
      return;
    }

    // Until popart supports more optimizers we only support SGD.
    type = OptimizerType::SGD;

    auto itr = opts.find("lr");
    ERROR_ON_MSG(itr == opts.end(),
                 "Learning rate was not provided in optimizer dictionary!");

    learningRate = itr->second;
    momentum = FindInMapOrZero(opts, "momentum");
    weightDecay = FindInMapOrZero(opts, "weight_decay");
    dampening = FindInMapOrZero(opts, "dampening");
  }

  OptimizerType type;

  std::pair<float, bool> learningRate;
  std::pair<float, bool> momentum;
  std::pair<float, bool> weightDecay;
  std::pair<float, bool> dampening;
};

class Compiler;
class SessionOptions {
public:
  SessionOptions();
  SessionOptions(SessionOptions &&);
  ~SessionOptions();
  // Disable copy: Move only
  SessionOptions(const SessionOptions &) = delete;
  SessionOptions &operator=(const SessionOptions &) = delete;

  void AddStringOption(const char *option, const char *value);
  void AddUInt64Option(const char *option, std::uint64_t value);
  void AddBoolOption(const char *option, bool value);
  void AddDoubleOption(const char *option, double value);
  // Insert a string option in an option container (set / list / vector)
  void InsertStringOption(const char *option, const char *value);
  // Insert a key / value pair in an option map
  void InsertStringPairOption(const char *option, const char *key,
                              const char *value);

private:
  std::unique_ptr<detail::SessionOptionsImpl> impl;
  friend Compiler;
};

class Compiler {
public:
  Compiler(bool isTraining, const SessionOptions &options);
  ~Compiler();
  Compiler(Compiler &&compiler);

  poptorch::TensorId AddInputTensor(const char *type,
                                    const std::vector<std::int64_t> &dims);

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<double>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define STRING const char *
#define NONE
#define ARG(Type, Name) , Type Name
#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs)       \
  poptorch::TensorId function(                                                 \
      const std::vector<poptorch::TensorId> &inputs Args);

#include "SupportedOperations.inc.h"

#undef BODY_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef INT_VEC
#undef FLOAT_VEC
#undef FLOAT
#undef INT
#undef BOOL
#undef STRING

  poptorch::TensorId
  AddInitializedInputTensor(const char *name, const char *type,
                            const std::vector<std::int64_t> &dims, void *data);

  bool TensorIdIsValid(poptorch::TensorId id) const;

  std::vector<std::int64_t> GetSize(poptorch::TensorId id);

  poptorch::TensorId
  customOperation(const char *op,
                  const std::vector<poptorch::TensorId> &inputs);

  void AddOutputType(OutputType type);

  // This function marks |output| as being read back from the device by the
  // host. |anchor_mode| determines how frequently that should happen.
  // clang-format off
  // "ALL":  Will return all popart batches.
  // "SUM": Will return the sum of all popart batches (I.E device iterations)
  // "EVERYN": Will return every N batch
  // "FINAL": Will return the last batch only
  // clang-format on
  void AddOutputTensor(poptorch::TensorId output);

  void SetUpInputOp(poptorch::TensorId id, float *ptr,
                    const std::vector<std::int64_t> &dims);

  void SetUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                    const std::vector<std::int64_t> &dims);

  void SetUpInputOp(poptorch::TensorId id, std::int64_t *ptr,
                    const std::vector<std::int64_t> &dims);

  void SetUpOutputOp(poptorch::TensorId id, float *ptr,
                     const std::vector<std::int64_t> &dims);

  void SetUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void SetUpOutputOp(poptorch::TensorId id, bool *ptr,
                     const std::vector<std::int64_t> &dims);
  void SetActiveIpu(std::uint64_t id);

  void InitSession(const Optimizer &opt);

  // Write the weights into IPU memory from the pytorch tensor buffers in the
  // model.
  void CopyWeightsToDevice();

  // Read the weights from IPU memory into the pytorch tensor buffers.
  void CopyWeightsToHost();

  // Return the type of the given tensor.
  PopartTypes GetPopartType(poptorch::TensorId tensor) const;

  /*
   * Execute the compiled popart graph using poplar. An optimizer can be
   * provided to update the optimizer currently being run by the graph. If there
   * is nothing to update the optimizer will be set to OptimizerType::None
   * otherwise the new optimizer will be written to device.
   */
  void Run(const Optimizer &optimizationToUpdate);

  std::uint64_t BatchPerStep() const;

  // Return the PopART batch dimensions [DeviceIterations * ReplicationFactor *
  // GradientAccumulation]
  std::uint64_t PopartBatchDim() const;

  // Take the above and work out how much of it is being returned. ID must anbe
  // an anchor d the batch dim will be mutated depending on what the anchor is
  // returning.
  std::uint64_t PopartBatchDimForAnchor(poptorch::TensorId id) const;

  // Return a flat representation of the output types
  // For example: ( T0, T2, (T3, T4)) is represented as:
  // [ Tuple3, Tensor, Tensor, Tuple2, Tensor, Tensor ]
  const std::vector<OutputType> &OutputTypes() const;

private:
  std::unique_ptr<detail::CompilerImpl> impl;
};

} // namespace poptorch

#endif // POPART_COMPILER_H
