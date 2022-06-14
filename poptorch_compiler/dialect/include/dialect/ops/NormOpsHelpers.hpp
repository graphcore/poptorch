// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_CODEGEN_POPTORCH_DIALECT_OPS_NORMOPSHELPERS_H_
#define POPTORCH_CODEGEN_POPTORCH_DIALECT_OPS_NORMOPSHELPERS_H_

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir {
class Value;
} // namespace mlir

namespace poptorch_ir {

// Based on PyTorch's _check_layer_norm_inputs, but also handles shared
// functionality (between forward and backward) of adding to operands and
// segments.
std::tuple<size_t, size_t, std::vector<int64_t>>
checkAndGetLayerNormDimsOperandsAndSegs(
    std::vector<mlir::Value> &operands, std::vector<std::int32_t> &segments,
    const mlir::Value &input, const std::vector<std::int64_t> &normalized_shape,
    const mlir::Value &weight, const mlir::Value &bias);

} // namespace poptorch_ir

#endif // POPTORCH_CODEGEN_POPTORCH_DIALECT_OPS_NORMOPSHELPERS_H_
