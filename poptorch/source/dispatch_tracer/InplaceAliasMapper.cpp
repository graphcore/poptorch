// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <spdlog/fmt/fmt.h>

#include "InplaceAliasMapper.hpp"

namespace poptorch {

InplaceArgAliasMapper &InplaceArgAliasMapper::getInstance() {
  static InplaceArgAliasMapper instance;
  return instance;
}

void InplaceArgAliasMapper::registerInplaceArgId(
    const std::string &operator_name, std::size_t alias_arg_id) {

  std::string key =
      _namespace ? fmt::format("{}::{}", _namespace.value(), operator_name)
                 : operator_name;
  _operator_name_to_arg_id.emplace(key, alias_arg_id);
}

std::optional<std::size_t>
InplaceArgAliasMapper::getInplaceArg(const std::string &operator_name) {
  auto &operator_name_to_arg_id = getInstance()._operator_name_to_arg_id;
  const auto it = operator_name_to_arg_id.find(operator_name);
  if (it != operator_name_to_arg_id.end()) {
    return it->second;
  }
  return std::nullopt;
}

void InplaceArgAliasMapper::setNamespace(const std::string &p_namespace) {
  _namespace = p_namespace;
}

void InplaceArgAliasMapper::unsetNamespace() { _namespace = std::nullopt; }

InplaceArgAliasMapperInit::InplaceArgAliasMapperInit(
    void (*init_mapper)(InplaceArgAliasMapper &),
    const std::string &p_namespace) {
  auto &alias_mapper = InplaceArgAliasMapper::getInstance();
  alias_mapper.setNamespace(p_namespace);
  init_mapper(alias_mapper);
  alias_mapper.unsetNamespace();
}

INPLACE_ARG_MAPPER_IMPL(torch_scatter, mapper) {
  mapper.registerInplaceArgId("scatter_mul", 3);
  mapper.registerInplaceArgId("scatter_max", 3);
  mapper.registerInplaceArgId("scatter_min", 3);
}

} // namespace poptorch
