// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_LOWER_TO_POPLAR_GRAPH_CONVERTER_HPP_
#define POPTORCH_LOWER_TO_POPLAR_GRAPH_CONVERTER_HPP_

namespace mlir {
template <typename T> class OperationPass;
class ModuleOp;
class TimingScope;
class PassManager;
} // namespace mlir

namespace poplar {
class Target;
} // namespace poplar
namespace poptorch_ir {

class CompilerContext;
class NonRestartingMLIRTimer;

class IMLIRGraphConverter {
public:
  ~IMLIRGraphConverter() = default;
  // Populate the Poplar Graph & Sequence in the context.
  void convertGraph(mlir::ModuleOp &module, NonRestartingMLIRTimer &timer);

protected:
  virtual void addCustomPasses(mlir::PassManager &manager) = 0;
};
} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_GRAPH_CONVERTER_HPP_
