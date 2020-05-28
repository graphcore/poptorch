// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_H
#define POPART_COMPILER_H

#include <memory>
#include <string>
#include <vector>

namespace poptorch {

using TensorId = std::size_t;

namespace detail {
struct CompilerImpl;
}

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
#define NONE
#define ARG(Type, Name) , Type Name
#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(StrFunc, function, OnnxImpl, Args, BodyArgs)                   \
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

  poptorch::TensorId
  AddInitializedInputTensor(const char *name, const char *type,
                            const std::vector<std::int64_t> &dims, void *data);

  std::vector<std::int64_t> GetSize(poptorch::TensorId id);

  poptorch::TensorId
  customOperation(const std::string &op,
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

  void SetActiveIpu(std::uint64_t id);

  void InitSession(bool profile);

  void Run();

  std::uint64_t BatchPerStep() const;

  std::uint64_t PopartBatchDim() const;

private:
  std::unique_ptr<detail::CompilerImpl> impl;
};

} // namespace poptorch

#endif // POPART_COMPILER_H
