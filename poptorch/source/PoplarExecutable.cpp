// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>

#include <iostream>
#include <sstream>
#include <string>

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/InplaceOps.hpp"
#include "poptorch/PoplarExecutable.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

void PoplarExecutable::updateOptimizers(
    const std::vector<Optimizer> &optimizers) {
  _compiler.updateOptimizers(optimizers);
}

std::vector<at::IValue>
PoplarExecutable::run(std::vector<at::Tensor> *inTensors) {
  std::vector<at::Tensor> tensor_views;

  // Set up the input tensors in the poplar graph to point to the incoming
  // pytorch tensors.
  for (std::size_t i = 0; i < _popart_inputs.size(); ++i) {
    poptorch::TensorId popart_id = _popart_inputs[i];
    at::Tensor &pytorch_tensor = inTensors->at(i);

    ERROR_ON(!pytorch_tensor.is_contiguous());

    // Convert to correct data type.
    std::vector<std::int64_t> popart_dims(pytorch_tensor.sizes().size());
    std::transform(pytorch_tensor.sizes().begin(), pytorch_tensor.sizes().end(),
                   popart_dims.begin(), [](std::int64_t j) { return j; });

    // Handle input based on the PyTorch input type
    at::ScalarType elem_type = pytorch_tensor.scalar_type();

    void *data_ptr = nullptr;
    if (pytorch_tensor.is_cpu()) {
      data_ptr = pytorch_tensor.data_ptr();
    } else {
      data_ptr = getDataSource(pytorch_tensor);
    }
    ERROR_ON(data_ptr == nullptr);

    switch (elem_type) {
    case at::ScalarType::Byte:
      _compiler.setUpInputOp(popart_id, static_cast<std::uint8_t *>(data_ptr),
                             popart_dims);
      break;
    case at::ScalarType::Char:
      _compiler.setUpInputOp(popart_id, static_cast<std::int8_t *>(data_ptr),
                             popart_dims);
      break;
    case at::ScalarType::Float:
      _compiler.setUpInputOp(popart_id, static_cast<float *>(data_ptr),
                             popart_dims);
      break;
    case at::ScalarType::Half:
      _compiler.setUpInputOp(popart_id, static_cast<std::int16_t *>(data_ptr),
                             popart_dims, true);
      break;
    case at::ScalarType::Short:
      _compiler.setUpInputOp(popart_id, static_cast<std::int16_t *>(data_ptr),
                             popart_dims);
      break;
    case at::ScalarType::Int:
      _compiler.setUpInputOp(popart_id, static_cast<std::int32_t *>(data_ptr),
                             popart_dims);
      break;
    case at::ScalarType::Bool:
      _compiler.setUpInputOp(popart_id, static_cast<bool *>(data_ptr),
                             popart_dims);
      break;
    case at::ScalarType::Long:
      // If it's an IPU tensor then it should have been handled by the
      // dispatcher.
      ERROR_ON_MSG(!pytorch_tensor.is_cpu(), "Only supported for CPU tensors");
      _converted_inputs[i] = pytorch_tensor.toType(at::ScalarType::Int);
      _compiler.setUpInputOp(
          popart_id,
          static_cast<std::int32_t *>(_converted_inputs[i].data_ptr()),
          popart_dims);
      break;
    case at::ScalarType::Double:
    case at::ScalarType::BFloat16:
      // If it's an IPU tensor then it should have been handled by the
      // dispatcher.
      ERROR_ON_MSG(!pytorch_tensor.is_cpu(), "Only supported for CPU tensors");
      _converted_inputs[i] = pytorch_tensor.toType(at::ScalarType::Float);
      _compiler.setUpInputOp(
          popart_id, static_cast<float *>(_converted_inputs[i].data_ptr()),
          popart_dims);
      break;
    default:
      ERROR("Unsupported input type torch."
            << torch::getTHPDtype(elem_type)->name);
    }
  }

  // Temp buffers for the output state.
  std::vector<at::IValue> returnees;
  returnees.reserve(_popart_outputs.size());

  // Set up the outputs.
  for (size_t i = 0; i < _popart_outputs.size(); i++) {
    poptorch::TensorId &popart_id(_popart_outputs[i]);
    std::vector<std::int64_t> dims = _compiler.getSize(popart_id);

    std::uint64_t b_dim = _compiler.popartBatchDimForAnchor(popart_id);
    if (b_dim > 1) {
      // Treat scalars as 1D tensors if necessary for batching.
      if (dims.empty()) {
        dims.push_back(1);
      }
      // Adjust by the popart batch dim, accounting for the anchor.
      dims[0] *= b_dim;
    }

    // Create the torch tensor and use its memory for the popart tensor.
    at::ScalarType type = _popart_output_types[i];
    returnees.emplace_back(at::empty(
        {dims}, at::dtype(type).memory_format(c10::MemoryFormat::Contiguous)));

    auto *data_ptr = returnees.back().toTensor().data_ptr();

    switch (type) {
    case at::ScalarType::Byte:
      _compiler.setUpOutputOp(popart_id, static_cast<std::uint8_t *>(data_ptr),
                              dims);
      break;
    case at::ScalarType::Char:
      _compiler.setUpOutputOp(popart_id, static_cast<std::int8_t *>(data_ptr),
                              dims);
      break;
    case at::ScalarType::Float:
      _compiler.setUpOutputOp(popart_id, static_cast<float *>(data_ptr), dims);
      break;
    case at::ScalarType::Half:
    case at::ScalarType::Short:
      _compiler.setUpOutputOp(popart_id, static_cast<std::int16_t *>(data_ptr),
                              dims);
      break;
    case at::ScalarType::Int:
      _compiler.setUpOutputOp(popart_id, static_cast<std::int32_t *>(data_ptr),
                              dims);
      break;
    case at::ScalarType::Bool:
      _compiler.setUpOutputOp(popart_id, static_cast<bool *>(data_ptr), dims);
      break;
    default:
      ERROR("Unexpected type returned from popart");
    }
  }

  // Execute the compiled poplar graph.
  _compiler.run();

  const auto &mapping = _inplace_info.input_output_mapping;
  for (size_t i = 0; i < mapping.size(); i++) {
    if (mapping[i] == InplaceGraphInfo::no_mapping) {
      continue;
    }
    auto out_tensor = returnees.at(mapping[i]).toTensor();
    (*inTensors)[i].copy_(out_tensor, false);
  }

  returnees.resize(_inplace_info.num_tensor_outputs);

  return returnees;
}

void PoplarExecutable::loadEngineAndConnectStreams() {
  if (!_compiler.isAttachedToDevice()) {
    _compiler.attachToDevice();
  }
  _compiler.loadEngineAndConnectStreams();
}

// Tell popart to copy weights off the IPU and write into host memory.
void PoplarExecutable::copyWeightsToHost(
    const std::map<std::string, void *> &buffers) {
  std::vector<void *> pointers;
  for (const std::string &name : _parameter_names) {
    pointers.push_back(buffers.at(name));
  }
  _compiler.copyWeightsToHost(pointers);
}

// Tell popart to copy weights from host into IPU memory.
void PoplarExecutable::copyWeightsToDevice(
    const std::map<std::string, void *> &buffers) {
  std::vector<void *> pointers;
  for (const std::string &name : _parameter_names) {
    pointers.push_back(buffers.at(name));
  }
  _compiler.copyWeightsToDevice(pointers);
}

const std::vector<OutputType> &PoplarExecutable::outputTypes() const {
  return _compiler.outputTypes();
}

std::string PoplarExecutable::getPopartIR() const {
  auto managed_ptr = _compiler.getPopartIR();

  const char *raw_ptr = static_cast<const char *>(managed_ptr.get());

  // Convert to std::string, copying again.
  return raw_ptr;
}

void PoplarExecutable::detachFromDevice() { _compiler.detachFromDevice(); }

void PoplarExecutable::attachToDevice() { _compiler.attachToDevice(); }

bool PoplarExecutable::isAttachedToDevice() const {
  return _compiler.isAttachedToDevice();
}

} // namespace poptorch
