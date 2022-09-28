// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

#include "../CompilerHelpers.hpp"

namespace poptorch_ir {

void copy_from_host::lowerToPoplar(CompilerContext &context) {
  const std::string ref = this->handle().str();
  poplar::Tensor output = context.fromSsa(this->result());
  poplar::DataStream fifo;

  auto itr = context.streams.find(ref);
  if (itr == context.streams.end()) {
    fifo = context.graph.addHostToDeviceFIFO(ref, output.elementType(),
                                             output.numElements());

    context.streams.insert({ref, fifo});
  } else {
    fifo = itr->second;
  }

  // Copy into the tensor.
  context.seq.add(poplar::program::Copy(fifo, output));
}

void copy_to_host::lowerToPoplar(CompilerContext &context) {
  const std::string ref = this->handle().str();
  poplar::Tensor input = context.fromSsa(this->tensor());
  poplar::DataStream fifo;

  auto itr = context.streams.find(ref);
  if (itr == context.streams.end()) {
    fifo = context.graph.addDeviceToHostFIFO(ref, input.elementType(),
                                             input.numElements());

    context.streams.insert({ref, fifo});
  } else {
    fifo = itr->second;
  }

  // Copy into the fifo.
  context.seq.add(poplar::program::Copy(input, fifo));
}

void copy_to_global_state::lowerToPoplar(CompilerContext &context) {
  auto tensor = context.fromSsa(this->tensor());
  auto global = context.fromSymbol(this->handle(), this->tensor().getType());

  if (global != tensor) {
    context.seq.add(poplar::program::Copy(tensor, global));
  }
}

void copy_from_global_state::lowerToPoplar(CompilerContext &context) {
  auto global = context.fromSymbol(this->handle(), this->tensor().getType());

  context.addTensor(this->tensor(), global, true);
}

} // namespace poptorch_ir
