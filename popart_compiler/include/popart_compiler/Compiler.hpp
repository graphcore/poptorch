// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_H
#define POPART_COMPILER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

using TensorId = std::size_t;

namespace detail {
struct CompilerImpl;
}

enum OptimizerType : std::uint8_t { NONE, SGD };

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

class Compiler {
public:
  Compiler(bool isTraining, std::uint64_t steps,
           std::uint64_t replicationFactor, std::uint64_t gradientAccumulation);
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

  std::vector<std::int64_t> GetSize(poptorch::TensorId id);

  poptorch::TensorId
  customOperation(const char *op,
                  const std::vector<poptorch::TensorId> &inputs);

  void AddOutput(poptorch::TensorId output);

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

  void SetActiveIpu(std::uint64_t id);

  void InitSession(bool profile, const Optimizer &opt);

  /*
   * Execute the compiled popart graph using poplar. An optimizer can be
   * provided to update the optimizer currently being run by the graph. If there
   * is nothing to update the optimizer will be set to OptimizerType::None
   * otherwise the new optimizer will be written to device.
   */
  void Run(const Optimizer &optimizationToUpdate);

  std::uint64_t BatchPerStep() const;

  std::uint64_t PopartBatchDim() const;

private:
  std::unique_ptr<detail::CompilerImpl> impl;
};

} // namespace poptorch

#endif // POPART_COMPILER_H
