// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <llvm/ADT/SmallVector.h>

#include "CompilerImpl.hpp"
#include "ExecutorImpl.hpp"
#include "pytorch_bridge/Compiler.hpp"
#include <iostream>
#include <utility>

namespace poptorch_ir {

PoptorchCompiler::PoptorchCompiler() {}
PoptorchCompiler::~PoptorchCompiler() {}

void PoptorchCompiler::init() {
  impl = std::make_unique<detail::PoptorchCompilerImpl>();
}

TensorId
PoptorchCompiler::addInput(void *,
                           const std::vector<std::int64_t> & shape,
                           Type type, const char *name) {
  mlir::RankedTensorType tensor = impl->getTensor(type, shape);

  // Add the argument to the function args.
  mlir::Value val = impl->addArgument(impl->main_graph, tensor);

  poptorch_ir::copy_from_host copy =
      impl->builder.create<poptorch_ir::copy_from_host>(impl->default_loc,
                                                          val, name);
  impl->main_graph.front().push_back(copy);

  impl->value_map.push_back(val);
  impl->input_callbacks.push_back(name);

  return impl->value_map.size() - 1;
}

TensorId
PoptorchCompiler::addParameter(void *ptr,
                               const std::vector<std::int64_t> & shape,
                               Type type, const char *name) {
  mlir::RankedTensorType tensor = impl->getTensor(type, shape);

  // Add the argument to the function args. Add to the main graph and both
  // weight copy graphs.
  mlir::Value val = impl->addArgument(impl->main_graph, tensor);

  // Write weights to the graph.
  poptorch_ir::copy_from_host copy_from =
      impl->builder.create<poptorch_ir::copy_from_host>(
          impl->default_loc, val, "Write-" + std::string(name));
  impl->write_weights_graph.front().push_back(copy_from);

  // Read weights from the graph.
  poptorch_ir::copy_to_host copy_to =
      impl->builder.create<poptorch_ir::copy_to_host>(
          impl->default_loc, val, "Read-" + std::string(name));
  impl->read_weights_graph.front().push_back(copy_to);

  // Add the main graph reference. Both copies are implicitly updating the main
  // graph reference.
  impl->value_map.push_back(val);

  impl->weight_callbacks.push_back({name, ptr});
  return impl->value_map.size() - 1;
}

TensorId PoptorchCompiler::addBuffer(void *ptr, const std::vector<std::int64_t>& shape,
                                     Type type, const char *name) {
  return addParameter(ptr, std::move(shape), type, name);
}

void PoptorchCompiler::addOutput(TensorId id, void *ptr,
                                        const char *name) {
  mlir::Value val = impl->value_map[id];
  poptorch_ir::copy_to_host copy =
      impl->builder.create<poptorch_ir::copy_to_host>(impl->default_loc,
                                                        val, name);
  impl->main_graph.front().push_back(copy);
  impl->output_callbacks.push_back({name, ptr});
}

void PoptorchCompiler::startTraceTiming() {
  impl->timing_manager.start("PytorchTracingTime");

  // TODO: Renable in LLVM 13
  /*
  impl->timing_manager.setEnabled(true);
  impl->root_timer_ = impl->timing_manager.getRootTimer();
  impl->tracer_timer_ = impl->root_timer.nest("PytorchTracingTime");
  */
}

void PoptorchCompiler::endTraceTiming() {
  impl->timing_manager.stop();

  // TODO: Renable in LLVM 13
  // We have to manually stop it anyway.
  // impl->tracer_timer.stop();
}

void PoptorchCompiler::getTimingInfo() {
  std::cout << impl->timing_manager.str(0) << std::endl;
  // TODO: Renable in LLVM 13
  // impl->timing_manager.dumpAsTree();
}

void PoptorchCompiler::dump() { impl->the_module.dump(); }

bool PoptorchCompiler::isView(poptorch_ir::TensorId id) const {
  mlir::Operation *op = impl->value_map[id].getDefiningOp();

  return op->hasTrait<mlir::OpTrait::ViewOp>();
}

std::vector<std::int64_t> PoptorchCompiler::getSize(TensorId id) const {
  mlir::Value val = impl->value_map[id];
  mlir::Type t1 = val.getType();

  //
  mlir::RankedTensorType t1_tensor = t1.cast<mlir::RankedTensorType>();
  return t1_tensor.getShape();
}

void PoptorchCompiler::addReturn() {
  // Add returns to each of the graphs.
  {
    poptorch_ir::end_graph ret =
        impl->builder.create<poptorch_ir::end_graph>(impl->default_loc);
    impl->main_graph.front().push_back(ret);
  }
  {
    poptorch_ir::end_graph ret =
        impl->builder.create<poptorch_ir::end_graph>(impl->default_loc);
    impl->write_weights_graph.front().push_back(ret);
  }
  {
    poptorch_ir::end_graph ret =
        impl->builder.create<poptorch_ir::end_graph>(impl->default_loc);
    impl->read_weights_graph.front().push_back(ret);
  }
}

} // namespace poptorch_ir
