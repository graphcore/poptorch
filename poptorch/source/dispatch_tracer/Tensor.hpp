// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TENSOR_HPP_
#define POPTORCH_DISPATCH_TENSOR_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/IpuSession.hpp"

namespace poptorch_ir {
class IIpuSession;
}

namespace poptorch {

using Buffer = poptorch_ir::Buffer;
using IpuTensorId = uint64_t;

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
  IpuTensorDetails(IpuTensorId tensor_id_, poptorch_ir::TensorType type_)
      : tensor_id(tensor_id_), type(std::move(type_)) {}

  const IpuTensorId tensor_id;
  const poptorch_ir::TensorType type;

  // For inputs that are temporaries we need the buffer to live until the
  // function is ran and we don't want to extend the lifetime of the
  // IpuTensorDetails unnecessarily. This means we need to share ownership of
  // the buffer
  std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>();
};

poptorch_ir::Type toCompilerType(const at::ScalarType &elem_type);
poptorch_ir::Type toCompilerElementType(const at::Tensor &tensor);
poptorch_ir::TensorType getTensorType(const at::Tensor &tensor);

uint64_t tensorImplDataSize(const at::TensorImpl &impl);

// Return the data size in bytes of the given at::Tensor.
uint64_t tensorDataSize(const at::Tensor &tensor);

// Return the tensor ID of the given IPU tensor.
IpuTensorId ipuTensorId(const at::Tensor &tensor);

// Return the tensor ID of the given IPU tensor implementation.
IpuTensorId ipuTensorId(const at::TensorImpl &tensor);

// Return true if the given at::Tensor is an IPU tensor.
bool isIpuTensor(const at::Tensor &tensor);

// Return a string containing the given tensor's metadata (device, shape, etc).
std::string str(const at::Tensor &tensor);

// Returns a reference to the CPU buffer of the given IPU tensor.
Buffer &getHostBuffer(const at::Tensor &ipu_tensor);

// Returns a reference to the CPU buffer of the given IPU tensor implementation.
Buffer &getHostBuffer(const at::TensorImpl &ipu_tensor);

std::shared_ptr<IpuTensorDetails>
getTensorDetails(const at::Tensor &ipu_tensor);

void errorOnZeroSizedTensor(const at::Tensor &tensor);

/** Host-side storage for `ipu` tensors.
 *
 *  This allows the user to convert tensors and modules to `ipu` using
 *  `t.to("ipu")` even when the dispatcher is off, and even outside eager mode.
 *
 *  We simply copy the tensor in to our ownership, then when we go to load and
 *  execute an executable, we can upload these tensors to the device. We'll
 *  also retrieve them from the device when the user copies a tensor back to the
 *  CPU (`t.to("cpu")`).
 */
class TensorStore {
public:
  TensorStore();
  TensorStore(const TensorStore &) = delete;
  TensorStore(TensorStore &&) = delete;
  TensorStore &operator=(TensorStore &) = delete;
  TensorStore &operator=(TensorStore &&) = delete;

  // Create a new IPU tensor.
  at::Tensor
  allocateTensor(c10::IntArrayRef sizes,
                 c10::optional<at::ScalarType> dtype = c10::nullopt,
                 c10::optional<at::Device> device = c10::nullopt,
                 c10::optional<at::Layout> layout = c10::nullopt,
                 c10::optional<bool> pin_memory = c10::nullopt,
                 c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

  void allocateBuffer(const at::Tensor &ipu_tensor);

  void copyOnIpu(const at::Tensor &ipu_dest, const at::Tensor &ipu_src);
  void copyFromCpu(const at::Tensor &ipu_dest, const at::Tensor &cpu_src);
  void copyToCpu(const at::Tensor &cpu_dest, const at::Tensor &ipu_src);

  void enableEagerMode();
  const std::shared_ptr<poptorch_ir::IIpuSession> &getIpuSession() const;

  void reset();

private:
  void allocateBuffer(IpuTensorDetails &details);

  poptorch_ir::TensorId _next_tensor_id{1};
  std::shared_ptr<poptorch_ir::IIpuSession> _ipu_session =
      poptorch_ir::createStaticSession();
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_TENSOR_HPP_
