// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

/*
 * Host op represents an operation executed on the CPU. It is offloaded by
 * writing the tensors from IPU into host buffers. Triggering the operation.
 * Then writing back to IPU tensors.
 */

#include <cstdint>
#include <popart/op.hpp>

extern "C" {

namespace poptorch_custom_ops {

constexpr std::uint32_t domain = 1;

// The number of input tensors we can consume (between MIN_INPUTS and
// MAX_INPUTS).
constexpr std::uint32_t min_inputs = 0;
constexpr std::uint32_t max_inputs = 64;

const popart::OperatorIdentifier host_op = {"poptorch.custom_ops",
                                            "HostOp",
                                            domain,
                                            {min_inputs, max_inputs}}; // NOLINT
} // namespace poptorch_custom_ops
}
