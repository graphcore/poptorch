// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_IPU_SESSION_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_IPU_SESSION_HPP_

#include <iterator>
#include <memory>
#include <variant>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/DebugInfo.hpp"
#include <poptorch_logging/Error.hpp>

namespace popit {
using Mem_t = struct MemRef;
} // namespace popit

namespace poptorch_ir {

struct FunctionIO {
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
};

class EagerIpuSession;
class HeadlessIpuSession;

class PopitMemPtr : public std::shared_ptr<popit::Mem_t> {
public:
  PopitMemPtr(std::nullptr_t);

private:
  // Only constructible from the eager ipu session
  friend class EagerIpuSession;
  friend class HeadlessIpuSession;
  explicit PopitMemPtr(std::shared_ptr<popit::Mem_t> ptr);
};

class Buffer {
  // TODO(T70841): since Buffer is stored as a shared pointer it should be
  // possible at least stop CpuBuffer being a shared pointer and it might be
  // possible to tidy up PopitMemPtr at the same time
  std::variant<std::monostate, CpuBuffer, PopitMemPtr> _store =
      std::monostate{};

public:
  Buffer() = default;
  explicit Buffer(CpuBuffer buf) noexcept;
  explicit Buffer(PopitMemPtr buf) noexcept;

  Buffer &operator=(CpuBuffer buf) noexcept;
  Buffer &operator=(PopitMemPtr buf) noexcept;

  const CpuBuffer &getCpuData();
  const CpuBuffer &getCpuData() const;

  PopitMemPtr &getPopitData();
  const PopitMemPtr &getPopitData() const;

  bool hasData() const;
};

class IIpuSession {
public:
  virtual ~IIpuSession() = default;

  virtual Buffer allocate(const TensorType &type) = 0;
  virtual void copyDataFromCpuSource(Buffer &ipu_dest, const char *cpu_src) = 0;
  virtual void copyDataToCpu(char *cpu_dest, Buffer &ipu_src) = 0;
  virtual void copyDataOnDevice(Buffer &dest, const Buffer &src) = 0;
};

std::shared_ptr<IIpuSession> createStaticSession();
std::shared_ptr<IIpuSession> createEagerSession(bool headless = false);

class PopitDeviceFunction;

class IAllocationMap {
public:
  virtual ~IAllocationMap() = default;

  virtual popit::Mem_t *getAllocation(TensorId id) const = 0;
  virtual popit::Mem_t *getOrAllocate(TensorId id, TensorDebugInfo info) = 0;
};

class PopitDeviceFunctionWrapper {
public:
  PopitDeviceFunctionWrapper(std::shared_ptr<PopitDeviceFunction> func,
                             FunctionIO io, GraphDebugInfo debug_info);
  PopitDeviceFunctionWrapper(PopitDeviceFunctionWrapper &&) noexcept = default;
  PopitDeviceFunctionWrapper &
  operator=(PopitDeviceFunctionWrapper &&) noexcept = default;
  PopitDeviceFunctionWrapper(const PopitDeviceFunctionWrapper &) = delete;
  PopitDeviceFunctionWrapper &
  operator=(const PopitDeviceFunctionWrapper &) = delete;

  ~PopitDeviceFunctionWrapper();

  void run(IAllocationMap &alloc_map) const;

private:
  std::shared_ptr<PopitDeviceFunction> _func;
  FunctionIO _io;
  GraphDebugInfo _debug_info;
};

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_IPU_SESSION_HPP_
