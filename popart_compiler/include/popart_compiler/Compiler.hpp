#ifndef POPART_COMPILER_H
#define POPART_COMPILER_H

#include <memory>

namespace poptorch {

using TensorId = std::size_t;

namespace detail {
struct CompilerImpl;
}

class Compiler {
public:
  Compiler();
  ~Compiler();

  poptorch::TensorId AddInputTensor(const char *type,
                                    const std::vector<std::int64_t> &dims);

  poptorch::TensorId BuildOp(const char *operation,
                             const std::vector<poptorch::TensorId> &inputs);

  poptorch::TensorId
  AddInitializedInputTensor(const char *name, const char *type,
                            const std::vector<std::int64_t> &dims, void *data);

  std::vector<std::int64_t> GetSize(poptorch::TensorId id);

  void AddOutput(poptorch::TensorId output);

  void SetUpInputOp(poptorch::TensorId id, void *ptr,
                    const std::vector<std::int64_t> &dims);

  void SetUpOutputOp(poptorch::TensorId id, void *ptr,
                    const std::vector<std::int64_t> &dims);


  void InitSession();


  void Run();


private:
  std::unique_ptr<detail::CompilerImpl> impl;
};

} // namespace poptorch

#endif // POPART_COMPILER_H
