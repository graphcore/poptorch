// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_POPIT_SESSION_HPP_
#define POPTORCH_POPIT_SESSION_HPP_

#include <memory>

#include "pytorch_bridge/DebugInfo.hpp"
#include "pytorch_bridge/IpuSession.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>

namespace mlir {
class ModuleOp;
}

namespace popit {
class Device;
struct Session;
using Session_t = Session;
} // namespace popit

namespace poptorch_ir {

class NonRestartingMLIRTimer;

class PopitFunctionCache final {
public:
  PopitDeviceFunctionWrapper emplaceWrapped(const mlir::ModuleOp &graph,
                                            EagerIpuSession &session,
                                            FunctionIO io,
                                            GraphDebugInfo debug_info,
                                            NonRestartingMLIRTimer &timer);

private:
  std::shared_ptr<PopitDeviceFunction> emplace(const mlir::ModuleOp &graph,
                                               EagerIpuSession &session,
                                               NonRestartingMLIRTimer &timer);
  llvm::DenseMap<llvm::hash_code, std::shared_ptr<PopitDeviceFunction>> _cache;
};

class EagerIpuSession final : public IIpuSession {
public:
  EagerIpuSession();
  ~EagerIpuSession();

  Buffer allocate(const TensorType &type) override;
  void copyDataFromCpuSource(Buffer &ipu_dest, const char *cpu_data) override;
  void copyDataToCpu(char *cpu_dest, Buffer &ipu_src) override;
  void copyDataOnDevice(Buffer &dest, const Buffer &src) override;

  // The popit session references the device. So the device needs to outlive the
  // session.
  std::unique_ptr<popit::Device> device;
  std::shared_ptr<popit::Session_t> session;
  PopitFunctionCache func_cache;
};

} // namespace poptorch_ir

#endif // POPTORCH_POPIT_SESSION_HPP_
