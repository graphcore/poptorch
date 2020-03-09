#include <poptorch/PoplarExecutable.hpp>
#include <iostream>
namespace poptorch {

at::IValue PoplarExecutable::Run(std::vector<at::Tensor> &inTensors) {

  // Set up the input tensors in the poplar graph to point to the incoming
  // pytorch tensors.
  for (std::size_t i = 0; i < popartInputs.size(); ++i) {
    poptorch::TensorId popartId = popartInputs[i];
    at::Tensor &pytorchTensor = inTensors[i];
    // Convert to correct data type.
    std::vector<std::int64_t> popartDims(pytorchTensor.sizes().size());

    std::transform(pytorchTensor.sizes().begin(), pytorchTensor.sizes().end(),
                   popartDims.begin(), [](std::int64_t i) { return i; });


    if (i == 0) {
        compiler.SetUpInputOp(
            popartId, static_cast<float *>(pytorchTensor.data_ptr()), popartDims);
    } else {
        compiler.SetUpInputOp(
            popartId, static_cast<std::int32_t *>(pytorchTensor.data_ptr()), popartDims);
    }
  }

  // Temp buffers for the output state.
  std::map<poptorch::TensorId, at::IValue> torchOutputs;

  // Set up the outputs.
  for (poptorch::TensorId id : popartOutputs) {
    std::vector<std::int64_t> dims = compiler.GetSize(id);

    dims[0] *= compiler.PopartBatchDim();

    // Create the torch tensor and use its memory for the popart tensor.
    torchOutputs[id] = at::empty({dims});
    float *dataPtr = (float *)torchOutputs[id].toTensor().data_ptr();

    compiler.SetUpOutputOp(id, dataPtr, dims);
  }

  // Execute the compiled poplar graph.
  compiler.Run();

  // Return the outputs as pytorch tensors to the user.
  for (auto &pair : torchOutputs) {
    return pair.second;
  }
}

} // namespace poptorch
