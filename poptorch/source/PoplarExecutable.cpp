// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/PoplarExecutable.hpp"
#include <iostream>
#include <string>

namespace poptorch {

std::vector<at::IValue>
PoplarExecutable::run(std::vector<at::Tensor> *inTensors,
                      const Optimizer &optimizer) {
  std::vector<at::Tensor> tensor_views;

  // Set up the input tensors in the poplar graph to point to the incoming
  // pytorch tensors.
  for (std::size_t i = 0; i < _popartInputs.size(); ++i) {
    poptorch::TensorId popart_id = _popartInputs[i];
    at::Tensor &pytorch_tensor = inTensors->at(i);

    ERROR_ON(!pytorch_tensor.is_contiguous());

    // Convert to correct data type.
    std::vector<std::int64_t> popart_dims(pytorch_tensor.sizes().size());
    std::transform(pytorch_tensor.sizes().begin(), pytorch_tensor.sizes().end(),
                   popart_dims.begin(), [](std::int64_t j) { return j; });

    at::ScalarType elem_type = pytorch_tensor.scalar_type();
    if (elem_type == at::ScalarType::Float) {
      _compiler.setUpInputOp(popart_id,
                             static_cast<float *>(pytorch_tensor.data_ptr()),
                             popart_dims);
    } else {
      at::Tensor as_int32 = pytorch_tensor.toType(at::ScalarType::Int);

      _compiler.setUpInputOp(popart_id,
                             static_cast<std::int32_t *>(as_int32.data_ptr()),
                             popart_dims);

      // Make sure the tensor doesn't get deallocated before IPU is invoked.
      tensor_views.push_back(as_int32);
    }
  }

  // Temp buffers for the output state.
  std::vector<at::IValue> returnees;
  returnees.reserve(_popartOutputs.size());

  // Set up the outputs.
  for (poptorch::TensorId id : _popartOutputs) {
    std::vector<std::int64_t> dims = _compiler.getSize(id);
    // Treat scalars as 1D tensors.
    if (dims.empty()) {
      dims.push_back(1);
    }

    // Adjust by the popart batch dim, accounting for the anchor.
    dims[0] *= _compiler.popartBatchDimForAnchor(id);

    poptorch::PopartTypes type = _compiler.getPopartType(id);

    // Create the torch tensor and use its memory for the popart tensor.
    if (type == poptorch::PopartTypes::FLOAT) {
      // Returned tensor is a tensor of floats.
      returnees.emplace_back(at::empty({dims}, at::ScalarType::Float));
      float *data_ptr =
          static_cast<float *>(returnees.back().toTensor().data_ptr());

      _compiler.setUpOutputOp(id, data_ptr, dims);
    } else if (type == poptorch::PopartTypes::INT32 ||
               type == poptorch::PopartTypes::UINT32) {
      // Return tensor is a tensor of ints.
      returnees.emplace_back(at::empty({dims}, at::ScalarType::Int));
      std::int32_t *data_ptr =
          static_cast<std::int32_t *>(returnees.back().toTensor().data_ptr());
      _compiler.setUpOutputOp(id, data_ptr, dims);
    } else if (type == poptorch::PopartTypes::BOOL) {
      // Return tensor is a tensor of bools.
      returnees.emplace_back(at::empty({dims}, at::ScalarType::Bool));
      bool *data_ptr =
          static_cast<bool *>(returnees.back().toTensor().data_ptr());
      _compiler.setUpOutputOp(id, data_ptr, dims);
    }
  }

  // Execute the compiled poplar graph.
  _compiler.run(optimizer);

  return returnees;
}

// Tell popart to copy weights off the IPU and write into host memory.
void PoplarExecutable::copyWeightsToHost() { _compiler.copyWeightsToHost(); }

// Tell popart to copy weights from host into IPU memory.
void PoplarExecutable::copyWeightsToDevice() {
  _compiler.copyWeightsToDevice();
}

const std::vector<OutputType> &PoplarExecutable::outputTypes() const {
  return _compiler.outputTypes();
}
} // namespace poptorch
