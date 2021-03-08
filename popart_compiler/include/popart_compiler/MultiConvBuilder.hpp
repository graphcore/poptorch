// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <popart/builder.hpp>
#include <popart/op/convbase.hpp>
#include <popart/tensors.hpp>

#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace detail {

class MultiConvBuilder {
public:
  void addConv(const std::vector<popart::TensorId> &inputs,
               const std::vector<int64_t> &dilations,
               const std::vector<int64_t> &kernel_shape,
               const std::vector<int64_t> &pads,
               const std::vector<int64_t> &strides) {
    // Record the inputs and attributes for this single conv
    _inputs.push_back(inputs);
    _dilations.push_back(dilations);
    _kernel_shape.push_back(kernel_shape);
    _pads.push_back(pads);
    _strides.push_back(strides);
  }

  void setAvailableMemoryProportions(const std::vector<float> &v) {
    _options.availableMemoryProportions = v;
  }

  void setPartialsTypes(const std::vector<int64_t> &partials_types) {
    std::vector<std::string> type_strs;

    for (int64_t t : partials_types) {
      if (t == 0) {
        type_strs.emplace_back("float");
      } else if (t == 1) {
        type_strs.emplace_back("half");
      } else {
        ERROR("Invalid MultiConv partials_types");
      }
    }

    _options.partialsTypes = type_strs;
  }

  void setPlanType(int64_t plan_type) {
    if (plan_type == 0) {
      _options.planType = "parallel";
    } else if (plan_type == 1) {
      _options.planType = "serial";
    } else {
      ERROR("Invalid MultiConv plan_type");
    }
  }

  void setPerConvReservedTiles(int n) { _options.perConvReservedTiles = n; }

  void setCycleBackOff(float v) { _options.cycleBackOff = v; }

  std::vector<popart::TensorId> build(popart::Builder *builder) const {
    auto opset = builder->aiGraphcoreOpset1();
    return opset.multiconv(_inputs, _dilations, {}, _pads, {}, _strides,
                           _options.availableMemoryProportions,
                           _options.partialsTypes, _options.planType,
                           _options.perConvReservedTiles,
                           _options.cycleBackOff);
  }

private:
  // Aggregated inputs for all the convs that are fused as a multiconv
  std::vector<std::vector<popart::TensorId>> _inputs;
  std::vector<std::vector<int64_t>> _dilations;
  std::vector<std::vector<int64_t>> _kernel_shape;
  std::vector<std::vector<int64_t>> _pads;
  std::vector<std::vector<int64_t>> _strides;
  popart::MultiConvOptions _options = {{}, {}};
};

} // namespace detail
} // namespace poptorch
