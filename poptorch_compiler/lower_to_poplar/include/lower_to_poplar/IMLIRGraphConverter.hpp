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
class NonRestartingMLIRTimer;

// Run the general, high-level optimisation and simplification passes on the
// MLIR graph.
void runGraphPasses(mlir::ModuleOp &module, NonRestartingMLIRTimer &timer);

class IMLIRGraphConverter {
public:
  void convertGraph(mlir::ModuleOp &module, NonRestartingMLIRTimer &timer);

protected:
  IMLIRGraphConverter() = default;
  virtual void addCustomPasses(mlir::PassManager &manager) = 0;
};

} // namespace poptorch_ir

#endif // POPTORCH_LOWER_TO_POPLAR_GRAPH_CONVERTER_HPP_
