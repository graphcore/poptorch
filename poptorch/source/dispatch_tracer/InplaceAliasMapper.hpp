// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_INPLACE_ALIAS_MAPPER_HPP_
#define POPTORCH_DISPATCH_INPLACE_ALIAS_MAPPER_HPP_

#include <c10/macros/Macros.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace poptorch {

class InplaceArgAliasMapper {
public:
  static InplaceArgAliasMapper &getInstance();
  static std::optional<std::size_t>
  getInplaceArg(const std::string &operator_name);

  void registerInplaceArgId(const std::string &operator_name,
                            std::size_t alias_arg_id);
  void setNamespace(const std::string &p_namespace);
  void unsetNamespace();

private:
  InplaceArgAliasMapper() = default;
  ~InplaceArgAliasMapper() = default;
  InplaceArgAliasMapper(const InplaceArgAliasMapper &) = delete;
  InplaceArgAliasMapper(InplaceArgAliasMapper &&) = delete;
  InplaceArgAliasMapper &operator=(const InplaceArgAliasMapper &) = delete;
  InplaceArgAliasMapper &operator=(InplaceArgAliasMapper &&) = delete;

  std::unordered_map<std::string, std::size_t> _operator_name_to_arg_id;
  std::optional<std::string> _namespace;
};

struct InplaceArgAliasMapperInit {
  InplaceArgAliasMapperInit(void (*init_mapper)(InplaceArgAliasMapper &),
                            const std::string &p_namespace);
};

#define INPLACE_ARG_MAPPER_IMPL(Namespace, mapper)                             \
  _INPLACE_ARG_MAPPER_IMPL(Namespace, mapper, C10_UID)

#define _INPLACE_ARG_MAPPER_IMPL(Namespace, mapper, uid)                       \
  static void Namespace##_##uid##_init_mapper_(InplaceArgAliasMapper &);       \
  static InplaceArgAliasMapperInit Namespace##_##uid##_init_arg_mapper =       \
      InplaceArgAliasMapperInit(&Namespace##_##uid##_init_mapper_,             \
                                #Namespace);                                   \
  static void Namespace##_##uid##_init_mapper_(InplaceArgAliasMapper &(mapper))

} // namespace poptorch

#endif // POPTORCH_DISPATCH_INPLACE_ALIAS_MAPPER_HPP_
