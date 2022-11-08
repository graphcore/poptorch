// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TENSOR_HPP_
#define POPTORCH_DISPATCH_TENSOR_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/DebugInfo.hpp"
#include "pytorch_bridge/IpuSession.hpp"

namespace poptorch_ir {
class IIpuSession;
}

namespace poptorch {

using Buffer = poptorch_ir::Buffer;
using IpuTensorId = uint64_t;

class IDispatch;
struct IpuTensorImpl;
class ValueMapper;

class ITensorView {
public:
  virtual poptorch_ir::TensorId addViewToGraph(IDispatch &dispatcher) = 0;
};

// The ipu tensor details tracks the data and meta information associated with
// the IpuTensorImpl. This information cannot be directly stored in the ipu
// tensor impl because the lifetime of that is too short when views are
// involved. We need to the lifetime of the data to outlive any views of the
// data.
struct IpuTensorDetails {
  IpuTensorDetails(IpuTensorId tensor_id_, poptorch_ir::TensorType type_,
                   std::shared_ptr<ITensorView> view_info);

  // The tensor details either owns its own storage or is a view of other tensor
  // details.
  //
  // For inputs that are temporaries we need the buffer to live until the
  // function is ran and we don't want to extend the lifetime of the
  // IpuTensorDetails unnecessarily. This means we need to share ownership of
  // the buffer.
  using Data =
      std::variant<std::shared_ptr<Buffer>, std::shared_ptr<ITensorView>>;

  const IpuTensorId tensor_id;
  const poptorch_ir::TensorType type;

  Data data;

  poptorch_ir::TensorDebugInfo debug_info;

  Buffer &getBuffer();
  std::shared_ptr<Buffer> getOwningBuffer() const;

  bool hasData() const;
  bool isView() const;
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

bool hasData(const at::Tensor &ipu_tensor);

std::shared_ptr<IpuTensorDetails>
getTensorDetails(const at::Tensor &ipu_tensor);

std::vector<std::shared_ptr<IpuTensorDetails>>
getTensorDetails(const std::vector<at::Tensor> &ipu_tensors);

void setTensorDetails(const at::Tensor &ipu_tensor,
                      std::shared_ptr<IpuTensorDetails> details);

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

  std::shared_ptr<IpuTensorDetails>
  allocateTensorDetails(c10::IntArrayRef size,
                        at::ScalarType coerced_scalar_type,
                        std::shared_ptr<ITensorView> view_info);
  // Create a new IPU tensor.
  at::Tensor allocateTensor(c10::IntArrayRef sizes,
                            c10::optional<at::ScalarType> dtype = c10::nullopt,
                            std::shared_ptr<ITensorView> view_info = nullptr,
                            c10::optional<at::Device> device = c10::nullopt);

  void allocateBuffer(const at::Tensor &ipu_tensor);

  void copyOnIpu(const at::Tensor &ipu_dest, const at::Tensor &ipu_src);
  void copyFromCpu(const at::Tensor &ipu_dest, const at::Tensor &cpu_src);
  void copyToCpu(const at::Tensor &cpu_dest, const at::Tensor &ipu_src);

  void enableEagerMode(bool headless);
  const std::shared_ptr<poptorch_ir::IIpuSession> &getIpuSession() const;

  void reset();

private:
  Buffer &allocateBuffer(IpuTensorDetails &details);

  poptorch_ir::TensorId _next_tensor_id{1};
  std::shared_ptr<poptorch_ir::IIpuSession> _ipu_session =
      poptorch_ir::createStaticSession();
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_TENSOR_HPP_
