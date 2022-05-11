// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TENSOR_HPP_
#define POPTORCH_DISPATCH_TENSOR_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <string>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch {

using Buffer = poptorch_ir::Buffer;

// Create an IPU at::Tensor.
at::Tensor createIpuTensor(at::ScalarType dtype, const at::Device &device,
                           uint64_t ipu_tensor_id, c10::IntArrayRef sizes,
                           c10::IntArrayRef strides);

// Mark an IPU tensor as being a parameter or not.
void setIsParameter(at::Tensor &tensor, bool is_parameter);

// Return true if the given IPU tensor is a parameter.
bool isParameter(const at::Tensor &tensor);

// Return true if the given IPU tensor implementation is a parameter.
bool isParameter(const at::TensorImpl &tensor);

// Return the data size in bytes of the given at::Tensor.
uint64_t tensorDataSize(const at::Tensor &tensor);

// Return the tensor ID of the given IPU tensor.
uint64_t ipuTensorId(const at::Tensor &tensor);

// Return true if the given at::Tensor is an IPU tensor.
bool isIpuTensor(const at::Tensor &tensor);

// Return a string containing the given tensor's metadata (device, shape, etc).
std::string str(const at::Tensor &tensor);

// Make a copy of the given CPU tensor's content and store it inside the given
// IPU tensor.
void copyDataFromCpuSource(at::Tensor &ipu_tensor, const at::Tensor &cpu_src);

// Return a reference to the cpu data of the given IPU tensor.
Buffer getCpuData(const at::Tensor &ipu_tensor);

// Return a reference to the cpu data of the given IPU tensor implementation.
Buffer getCpuData(const at::TensorImpl &ipu_tensor);

} // namespace poptorch

#endif // POPTORCH_DISPATCH_TENSOR_HPP_
