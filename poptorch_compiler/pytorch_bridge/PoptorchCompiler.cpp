// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PoptorchCompiler.hpp"

#include <llvm/ADT/SmallVector.h>

#include <iostream>
#include <utility>

#include "CompilerImpl.hpp"
#include "ExecutorImpl.hpp"

namespace poptorch_ir {

PoptorchCompiler::PoptorchCompiler() {}
PoptorchCompiler::~PoptorchCompiler() {}

void PoptorchCompiler::init() {
  _impl = std::make_unique<detail::PoptorchCompilerImpl>();
}

TensorId PoptorchCompiler::addInput(void * /*unused*/,
                                    const std::vector<std::int64_t> &shape,
                                    Type type, const char *name) {
  mlir::RankedTensorType tensor = _impl->getTensor(type, shape);

  // Add the argument to the function args.
  mlir::Value val = _impl->addArgument(_impl->main_graph, tensor);

  poptorch_ir::copy_from_host copy =
      _impl->builder.create<poptorch_ir::copy_from_host>(_impl->default_loc,
                                                         val, name);
  _impl->main_graph.front().push_back(copy);

  _impl->value_map.push_back(val);
  _impl->input_callbacks.push_back(name);

  return _impl->value_map.size() - 1;
}

TensorId PoptorchCompiler::addParameter(void *ptr,
                                        const std::vector<std::int64_t> &shape,
                                        Type type, const char *name) {
  mlir::RankedTensorType tensor = _impl->getTensor(type, shape);

  // Add the argument to the function args. Add to the main graph and both
  // weight copy graphs.
  mlir::Value val = _impl->addArgument(_impl->main_graph, tensor);

  // Write weights to the graph.
  poptorch_ir::copy_from_host copy_from =
      _impl->builder.create<poptorch_ir::copy_from_host>(
          _impl->default_loc, val, "Write-" + std::string(name));
  _impl->write_weights_graph.front().push_back(copy_from);

  // Read weights from the graph.
  poptorch_ir::copy_to_host copy_to =
      _impl->builder.create<poptorch_ir::copy_to_host>(
          _impl->default_loc, val, "Read-" + std::string(name));
  _impl->read_weights_graph.front().push_back(copy_to);

  // Add the main graph reference. Both copies are implicitly updating the main
  // graph reference.
  _impl->value_map.push_back(val);

  _impl->weight_callbacks.push_back({name, ptr});
  return _impl->value_map.size() - 1;
}

TensorId PoptorchCompiler::addBuffer(void *ptr,
                                     const std::vector<std::int64_t> &shape,
                                     Type type, const char *name) {
  return addParameter(ptr, shape, type, name);
}

void PoptorchCompiler::addOutput(TensorId id, void *ptr, const char *name) {
  mlir::Value val = _impl->value_map[id];
  poptorch_ir::copy_to_host copy =
      _impl->builder.create<poptorch_ir::copy_to_host>(_impl->default_loc, val,
                                                       name);
  _impl->main_graph.front().push_back(copy);
  _impl->output_callbacks.push_back({name, ptr});
}

void PoptorchCompiler::startTraceTiming() {
  _impl->timing_manager.start("PytorchTracingTime");

  // TODO(T49565): Renable in LLVM 13
  /*
  _impl->timing_manager.setEnabled(true);
  _impl->root_timer_ = _impl->timing_manager.getRootTimer();
  _impl->tracer_timer_ = _impl->root_timer.nest("PytorchTracingTime");
  */
}

void PoptorchCompiler::endTraceTiming() {
  _impl->timing_manager.stop();

  // TODO(T49565): Renable in LLVM 13
  // We have to manually stop it anyway.
  // _impl->tracer_timer.stop();
}

void PoptorchCompiler::getTimingInfo() {
  std::cout << _impl->timing_manager.str(0) << std::endl;
  // TODO(T49565): Renable in LLVM 13
  // _impl->timing_manager.dumpAsTree();
}

void PoptorchCompiler::dump() { _impl->the_module.dump(); }

bool PoptorchCompiler::isView(poptorch_ir::TensorId id) const {
  mlir::Operation *op = _impl->value_map[id].getDefiningOp();

  return op->hasTrait<mlir::OpTrait::ViewOp>();
}

std::vector<std::int64_t> PoptorchCompiler::getSize(TensorId id) const {
  mlir::Value val = _impl->value_map[id];
  mlir::Type t1 = val.getType();

  //
  mlir::RankedTensorType t1_tensor = t1.cast<mlir::RankedTensorType>();
  return t1_tensor.getShape();
}

void PoptorchCompiler::addReturn() {
  // Add returns to each of the graphs.
  {
    poptorch_ir::end_graph ret =
        _impl->builder.create<poptorch_ir::end_graph>(_impl->default_loc);
    _impl->main_graph.front().push_back(ret);
  }
  {
    poptorch_ir::end_graph ret =
        _impl->builder.create<poptorch_ir::end_graph>(_impl->default_loc);
    _impl->write_weights_graph.front().push_back(ret);
  }
  {
    poptorch_ir::end_graph ret =
        _impl->builder.create<poptorch_ir::end_graph>(_impl->default_loc);
    _impl->read_weights_graph.front().push_back(ret);
  }
}

} // namespace poptorch_ir
