// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "popart_compiler/PopartEnums.hpp"

// This header should contain ABI agnostic data types which are
// used to share data with other PopTorch components.
// Types in this file must not depend on external symbols.
namespace poptorch {
namespace popart_compiler {

// PopTorch abstraction of popart::MutableVoidData to be used across the ABI
// boundary
struct TensorMetadata {
  const char *id;
  std::vector<int64_t> shape;
  const char *dtype;
  void *data = nullptr;
  int64_t num_bytes = -1;
};

/*
  We use this callback structure to capture data from the poptorch python
  frontend. We get the function to call as well as pointers to the output/input
  storage waiting on CPU. From this we derive more data, see
  CallbackInternalMetadata in CompilerImpl.hpp.
*/
struct CallbackMetadata {
  // The thing we are calling back.
  std::function<void()> the_callback;

  // Due to tracing complexities we have to register the buffers as a seperate
  // step after the model has been traced.
  std::function<void()> buffer_registration_callback;

  // Pointers to the buffers we created on host.
  std::vector<void *> input_pointers;
  std::vector<void *> output_pointers;
};

using TensorId = std::size_t;

static constexpr TensorId NoneTensor = 0; // NOLINT

enum class OutputElemType { Tensor, Tuple, List };

// For testing only: throw an exception of the selected type.
enum class TestErrorType {
  Poptorch,
  Popart,
  PopartInternal,
  Poplibs,
  PoplarUnrecoverable,
  PoplarUnknown,
  PoplarRecoverableFullReset,
  PoplarLinkError
};

struct OutputTypeShape {
  OutputElemType type;
  int64_t num_elements{0};
};

struct Timestamps {
  std::vector<std::vector<double>> input;
  std::vector<std::vector<double>> input_complete;
  std::vector<std::vector<double>> output;
  std::vector<std::vector<double>> output_complete;
};

struct Optimizer {
  struct Parameter {
    char name[32];
    float value;
    bool is_const;
  };
  using ParamType = std::pair<float, bool>;

  explicit Optimizer(OptimizerType t, bool useTfVariant)
      : type(t), accum_types_provided(false), use_tf_variant(useTfVariant) {}
  explicit Optimizer(OptimizerType t, bool useTfVariant, float maxGradNorm)
      : type(t), accum_types_provided(false), use_tf_variant(useTfVariant),
        max_grad_norm(maxGradNorm) {}
  Optimizer(OptimizerType t, bool accumType, bool firstOrderType,
            bool secondOrderType, bool useTfVariant, float maxGradNorm)
      : type(t), accum_types_provided(true), accum_type_is_half(accumType),
        first_order_momentum_accum_type_is_half(firstOrderType),
        second_order_momentum_accum_type_is_half(secondOrderType),
        use_tf_variant(useTfVariant), max_grad_norm(maxGradNorm) {}

  OptimizerType type;
  // True if the main, first and second order accum types have been set.
  bool accum_types_provided;
  // Special parameters for adam/lamb. If true accumulations will be half
  // otherwise will be float.
  bool accum_type_is_half;
  bool first_order_momentum_accum_type_is_half;
  bool second_order_momentum_accum_type_is_half;
  bool use_tf_variant;
  float max_grad_norm;

  std::vector<Parameter> parameters;
};

} // namespace popart_compiler
} // namespace poptorch
