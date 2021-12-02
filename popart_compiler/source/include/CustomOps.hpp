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

extern const char host_op_metadata_attr[];
const popart::OperatorIdentifier host_op = {"poptorch.custom_ops",
                                            "HostOp",
                                            domain,
                                            {min_inputs, max_inputs}}; // NOLINT
const popart::OperatorIdentifier upsample_bilinear2d = {
    "poptorch.custom_ops", "UpsampleBilinear2d", 1};
const popart::OperatorIdentifier upsample_bilinear2d_grad = {
    "poptorch.custom_ops", "UpsampleBilinear2dGrad", 1};

const popart::OperatorIdentifier torch_softplus = {
    "poptorch.custom_ops", "TorchSoftplus", 1, {1}, 1};

const popart::OperatorIdentifier torch_softplus_inplace = {
    "poptorch.custom_ops", "TorchSoftplusInplace", 1, {1}, 1};

const popart::OperatorIdentifier torch_softplus_grad = {
    "poptorch.custom_ops", "TorchSoftplusGrad", 1, {1}, 1};

const popart::OperatorIdentifier embedding = {"poptorch.custom_ops",
                                              "Embedding", domain};

const popart::OperatorIdentifier embedding_grad = {"poptorch.custom_ops",
                                                   "EmbeddingGrad", domain};

} // namespace poptorch_custom_ops
}
