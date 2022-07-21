// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TENSOR_HPP_
#define POPTORCH_DISPATCH_TENSOR_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch {

using Buffer = poptorch_ir::Buffer;

struct IpuTensorImpl;
class ValueMapper;

// We cannot reliably store references to TensorImpls, as they may be deleted
// without notice if the tensor goes out of scope. We can't hold an
// intrusive_ptr or weak_intrusive_ptr to them either, as they may be
// ExclusivelyOwned, which will trigger an assert if an intrusive_ptr
// exists at the time of destruction. So instead, we store the details we
// care about in this structure, which is referenced by a shared_ptr from
// IpuTensorImpl and also from the ValueMapper.
struct IpuTensorDetails {
  // Raw pointer to the parent IpuTensorImpl. If the parent is destroyed,
  // the details of the tensor will be copied into this structure so they can
  // still be accessed through the ValueMapper.
  IpuTensorImpl *parent;
  ValueMapper *mapper = nullptr;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  bool is_parameter = false;
  Buffer host_buffer;

  int64_t dim();
  c10::IntArrayRef sizesArrayref();
  c10::IntArrayRef stridesArrayref();
  int64_t numel();
};

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

// Return the tensor ID of the given IPU tensor implementation.
uint64_t ipuTensorId(const at::TensorImpl &tensor);

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

std::shared_ptr<IpuTensorDetails>
getTensorDetails(const at::TensorImpl &ipu_tensor);

} // namespace poptorch

#endif // POPTORCH_DISPATCH_TENSOR_HPP_
