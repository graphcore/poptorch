// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/PopartEnums.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace popart {
enum class DataType;
} // namespace popart

namespace poptorch {

using TensorId = std::size_t;

namespace detail {
struct CompilerImpl;
struct SessionOptionsImpl;
} // namespace detail

struct OutputType {
  enum class Type { Tensor, Tuple, List };
  Type type;
  int64_t num_elements{0};
};

// Extract the value from the map or return zero.
static std::pair<float, bool> findInMapOrZero(
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

/* Returns true if the system contains a device with numIpus
 * Note: This function doesn't check if the devices are currently in use.
 */
bool ipuHardwareIsAvailable(std::uint64_t num_ipus = 1);

struct Optimizer {
  explicit Optimizer(
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

    learning_rate = itr->second;
    momentum = findInMapOrZero(opts, "momentum");
    weight_decay = findInMapOrZero(opts, "weight_decay");
    dampening = findInMapOrZero(opts, "dampening");
  }

  OptimizerType type;

  std::pair<float, bool> learning_rate;
  std::pair<float, bool> momentum;
  std::pair<float, bool> weight_decay;
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

  void addStringOption(const char *option, const char *value);
  void addUint64Option(const char *option, std::uint64_t value);
  void addBoolOption(const char *option, bool value);
  void addDoubleOption(const char *option, double value);
  // Insert a string option in an option container (set / list / vector)
  void insertStringOption(const char *option, const char *value);
  // Insert a key / value pair in an option map
  void insertStringPairOption(const char *option, const char *key,
                              const char *value);

private:
  std::unique_ptr<detail::SessionOptionsImpl> _impl;
  friend Compiler;
};

class Compiler {
public:
  Compiler(bool is_training, const SessionOptions &options);
  ~Compiler();
  Compiler(Compiler &&compiler);

  poptorch::TensorId addInputTensor(const char *type,
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

#include "SupportedOperations.inc.hpp"

#undef OP_DECL
#undef BODY_ARG
#undef ARG
#undef STRING
#undef NONE
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC

  poptorch::TensorId
  addInitializedInputTensor(const char *name, const char *type,
                            const std::vector<std::int64_t> &dims, void *data);

  bool tensorIdIsValid(poptorch::TensorId id) const;

  std::vector<std::int64_t> getSize(poptorch::TensorId id) const;

  poptorch::TensorId
  customOperation(const char *op,
                  const std::vector<poptorch::TensorId> &inputs);

  void addOutputType(OutputType type);

  // This function marks |output| as being read back from the device by the
  // host. |anchor_mode| determines how frequently that should happen.
  // clang-format off
  // "ALL":  Will return all popart batches.
  // "SUM": Will return the sum of all popart batches (I.E device iterations)
  // "EVERYN": Will return every N batch
  // "FINAL": Will return the last batch only
  // clang-format on
  void addOutputTensor(poptorch::TensorId output);

  void setUpInputOp(poptorch::TensorId id, float *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(poptorch::TensorId id, std::int16_t *ptr,
                    const std::vector<std::int64_t> &dims,
                    bool float16 = false);

  void setUpOutputOp(poptorch::TensorId id, float *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, bool *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(poptorch::TensorId id, std::int16_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void setActiveIpu(std::uint64_t id);

  void initSession(const Optimizer &opt);

  // Write the weights into IPU memory from the pytorch tensor buffers in the
  // model.
  void copyWeightsToDevice();

  // Read the weights from IPU memory into the pytorch tensor buffers.
  void copyWeightsToHost();

  // Return the type of the given tensor.
  PopartTypes getPopartType(poptorch::TensorId tensor) const;

  /*
   * Execute the compiled popart graph using poplar. An optimizer can be
   * provided to update the optimizer currently being run by the graph. If there
   * is nothing to update the optimizer will be set to OptimizerType::None
   * otherwise the new optimizer will be written to device.
   */
  void run(const Optimizer &optimizer);

  std::uint64_t batchPerStep() const;

  // Return the PopART batch dimensions [DeviceIterations * ReplicationFactor *
  // GradientAccumulation]
  std::uint64_t popartBatchDim() const;

  // Take the above and work out how much of it is being returned. ID must anbe
  // an anchor d the batch dim will be mutated depending on what the anchor is
  // returning.
  std::uint64_t popartBatchDimForAnchor(poptorch::TensorId id) const;

  // Return a flat representation of the output types
  // For example: ( T0, T2, (T3, T4)) is represented as:
  // [ Tuple3, Tensor, Tensor, Tuple2, Tensor, Tensor ]
  const std::vector<OutputType> &outputTypes() const;

private:
  void assertTensorIs(PopartTypes dataType, const poptorch::TensorId &id,
                      const char *caller) const;

  std::unique_ptr<detail::CompilerImpl> _impl;
};

} // namespace poptorch
