// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/PoplarExecutable.hpp"
#include <iostream>
#include <string>
#include <unordered_map>

namespace poptorch {

std::vector<at::IValue> PoplarExecutable::Run(
    std::vector<at::Tensor> &inTensors,
    const std::unordered_map<std::string, std::pair<float, bool>>
        &optimizerParameters) {
  std::vector<at::Tensor> tensor_views;

  // Set up the input tensors in the poplar graph to point to the incoming
  // pytorch tensors.
  for (std::size_t i = 0; i < popartInputs.size(); ++i) {
    poptorch::TensorId popartId = popartInputs[i];
    at::Tensor &pytorchTensor = inTensors[i];

    ERROR_ON(!pytorchTensor.is_contiguous());

    // Convert to correct data type.
    std::vector<std::int64_t> popartDims(pytorchTensor.sizes().size());
    std::transform(pytorchTensor.sizes().begin(), pytorchTensor.sizes().end(),
                   popartDims.begin(), [](std::int64_t j) { return j; });

    at::ScalarType elemType = pytorchTensor.scalar_type();
    if (elemType == at::ScalarType::Float) {
      compiler.SetUpInputOp(
          popartId, static_cast<float *>(pytorchTensor.data_ptr()), popartDims);
    } else {
      at::Tensor asInt32 = pytorchTensor.toType(at::ScalarType::Int);

      compiler.SetUpInputOp(popartId,
                            static_cast<std::int32_t *>(asInt32.data_ptr()),
                            popartDims);

      // Make sure the tensor doesn't get deallocated before IPU is invoked.
      tensor_views.push_back(asInt32);
    }
  }

  // Temp buffers for the output state.
  std::vector<at::IValue> returnees;
  returnees.reserve(popartOutputs.size());

  // Set up the outputs.
  for (poptorch::TensorId id : popartOutputs) {
    std::vector<std::int64_t> dims = compiler.GetSize(id);
    // Treat scalars as 1D tensors.
    if (dims.size() == 0) {
      dims.push_back(1);
    }

    // Adjust by the popart batch dim, accounting for the anchor.
    dims[0] *= compiler.PopartBatchDimForAnchor(id);

    poptorch::PopartTypes type = compiler.GetPopartType(id);

    // Create the torch tensor and use its memory for the popart tensor.
    if (type == poptorch::PopartTypes::FLOAT) {
      returnees.emplace_back(at::empty({dims}, at::ScalarType::Float));
      float *dataPtr =
          static_cast<float *>(returnees.back().toTensor().data_ptr());

      compiler.SetUpOutputOp(id, dataPtr, dims);
    } else if (type == poptorch::PopartTypes::INT32 ||
               type == poptorch::PopartTypes::UINT32) {
      returnees.emplace_back(at::empty({dims}, at::ScalarType::Int));
      std::int32_t *dataPtr =
          static_cast<std::int32_t *>(returnees.back().toTensor().data_ptr());
      compiler.SetUpOutputOp(id, dataPtr, dims);
    }
  }

  // Execute the compiled poplar graph.
  compiler.Run(optimizerParameters);

  return returnees;
}

// Tell popart to copy weights off the IPU and write into host memory.
void PoplarExecutable::CopyWeightsToHost() { compiler.CopyWeightsToHost(); }

// Tell popart to copy weights from host into IPU memory.
void PoplarExecutable::CopyWeightsToDevice() { compiler.CopyWeightsToDevice(); }

const std::vector<OutputType> &PoplarExecutable::OutputTypes() const {
  return compiler.OutputTypes();
}
} // namespace poptorch
