// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "popart_compiler/CompilerOptions.hpp"

namespace poptorch {
namespace detail {

struct SessionOptionsImpl {
  SessionOptionsImpl();

  std::map<std::string, std::function<void(bool)>> bool_options;
  std::map<std::string, std::function<void(std::uint64_t)>> uint64_options;
  std::map<std::string, std::function<void(std::string)>> string_options;
  std::map<std::string, std::function<void(double)>> double_options;

  std::map<std::string,
           std::function<void(std::pair<std::string, std::string>)>>
      container_options;
  std::set<std::string> options_set;

  popart::SessionOptions popart_options;
  CompilerOptions poptorch_options;

  void setMemoryProportion(std::uint32_t ipu, float memory) {
    poptorch_options.available_memory_proportion[ipu] = memory;
  }

  template <typename ValueType>
  void set(const std::string &key, ValueType value,
           std::map<std::string, std::function<void(ValueType)>> &options,
           const std::string &typeStr) {
    auto it = options.find(key);
    ERROR_ON_MSG(it == options.end(),
                 "Unknown " << typeStr << " option " << key);
    it->second(value);
    options_set.insert(key);
  }
};

} // namespace detail
} // namespace poptorch
