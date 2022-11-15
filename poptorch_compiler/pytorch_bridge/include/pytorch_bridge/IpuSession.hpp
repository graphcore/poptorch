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

namespace poptorch_ir {

struct FunctionIO {
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
};

class Buffer {
  // TODO(T70841): since Buffer is stored as a shared pointer it should be
  // possible at least stop CpuBuffer being a shared pointer.
  std::variant<std::monostate, CpuBuffer> _store = std::monostate{};

public:
  Buffer() = default;
  explicit Buffer(CpuBuffer buf) noexcept;

  Buffer &operator=(CpuBuffer buf) noexcept;

  const CpuBuffer &getCpuData();
  const CpuBuffer &getCpuData() const;

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

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_IPU_SESSION_HPP_
