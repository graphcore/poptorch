// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>

#include <popart/ir.hpp>

#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include <popart/names.hpp>
#include <popart/opidentifier.hpp>

#include <CustomOps.hpp>

#include "popart_compiler/CompilerImpl.hpp"
/*
 * A popart custom operation to handle Host operations. Takes a callback and
 * sets up the IPU->CPU communication for the needed tensors.
 */
namespace poptorch {

namespace {

// Get the popart type info for a given output from the stream metadata.
popart::TensorInfo
shapeInferOutput(detail::CallbackInternalMetadata *stream_info,
                 std::uint32_t i) {
  // Get type and shape from metadata.
  const popart::DataType type =
      poptorch::popartTypeFromPoptorch(stream_info->output_types[i]);
  const std::vector<std::size_t> &shape = stream_info->output_shapes[i];

  // Convert from the poptorch/poplar type (std::size_t) to the popart one
  // (std::uint64_t).
  popart::Shape as_popart_shape;
  as_popart_shape.reserve(shape.size());
  for (std::size_t elem : shape) {
    as_popart_shape.push_back(elem);
  }

  // Create popart info.
  return popart::TensorInfo{type, as_popart_shape};
}

detail::CallbackInternalMetadata *
getMetadataFromAttributeMap(const popart::Attributes &attrs) {
  // Pointer smuggled in via an integer.
  std::int64_t as_int =
      attrs.getAttribute<popart::Attributes::Int>("stream_info");

  logging::trace("Pointer retrieved by CPU op {}", as_int);

  std::intptr_t as_ptr = static_cast<std::intptr_t>(as_int);

  logging::trace("Casted from {} to {}", as_int, as_ptr);

  // Cast to the correct type.
  return reinterpret_cast<poptorch::detail::CallbackInternalMetadata *>(as_ptr);
}

} // namespace
/*
  Popart custom op which uses the metadata gathered by the compiler to setup
  poplar tensors and copy into/from them from/to host.
*/
class HostOp : public popart::Op {
public:
  HostOp(const popart::OperatorIdentifier &_opid,
         poptorch::detail::CallbackInternalMetadata *info,
         const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), stream_info(info) {}

  // Configure the output popart Tensor
  void setup() override {
    // Tell popart what the output should look like.
    for (std::uint32_t i = 0; i < stream_info->output_types.size(); ++i) {
      outInfo(i) = shapeInferOutput(stream_info, i);
    }
  }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<HostOp>(*this);
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  poptorch::detail::CallbackInternalMetadata *stream_info;
};

class HostOpx : public popart::popx::Opx {
public:
  HostOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<HostOp>(op, poptorch_custom_ops::host_op);

    stream_info = dynamic_cast<HostOp *>(op)->stream_info;
  }

  void grow(poplar::program::Sequence &sequence) const override {
    poplar::Graph &graph = this->graph();

    // Get basic op info from metadata.
    const std::uint32_t num_inputs = stream_info->input_handles.size();
    const std::uint32_t num_outputs = stream_info->output_handles.size();

    // For each input create the FIFO and copy from it into the poplar tensor
    // popart has already created/
    for (std::uint32_t input_index = 0; input_index < num_inputs;
         ++input_index) {
      const std::string input_handle = stream_info->input_handles[input_index];

      // poplar::Tensor from popart.
      poplar::Tensor input = getInTensor(input_index);

      // FIFO we can add a callback to in compiler.cpp.
      poplar::DataStream stream = graph.addDeviceToHostFIFO(
          input_handle, input.elementType(), input.numElements());

      // Host->IPU copy.
      sequence.add(poplar::program::Copy(input, stream));
    }

    // Force an IPU sync to prevent poplar from merging the stream copies from
    // above from getting merged with those below. We do this as the copies
    // above will call the python callback via the poplar callback handlers.
    sequence.add(poplar::program::Sync(poplar::SyncType::INTERNAL));

    for (std::uint32_t output = 0; output < num_outputs; ++output) {
      const std::string output_handle = stream_info->output_handles[output];

      const poplar::Type type =
          poptorch::poplarTypeFromPoptorch(stream_info->output_types[output]);

      const std::vector<std::size_t> &shape =
          stream_info->output_shapes[output];

      // Add the poplar tensor.
      poplar::Tensor output_tensor = graph.addVariable(
          type, shape, poplar::VariableMappingMethod::LINEAR, output_handle);

      // Count the number of elements.
      std::size_t number_of_elements = 1;
      for (std::size_t elem : shape) {
        number_of_elements = elem * number_of_elements;
      }

      // FIFO we can add a callback to in compiler.cpp.
      poplar::DataStream stream =
          graph.addHostToDeviceFIFO(output_handle, type, number_of_elements);

      // IPU->Host copy.
      sequence.add(poplar::program::Copy(stream, output_tensor));

      // Tell popart this is the output.
      setOutTensor(output, output_tensor);
    }
  }

  poptorch::detail::CallbackInternalMetadata *stream_info;
};

} // namespace poptorch

static popart::OpCreator<poptorch::HostOp> host_op_creator(
    {{poptorch_custom_ops::host_op, {}}},
    [](const popart::OpCreatorInfo &info) {
      // Get the stream info from the attribute map we passed to create the op.
      auto stream_info = poptorch::getMetadataFromAttributeMap(info.attributes);

      return std::unique_ptr<popart::Op>(
          new poptorch::HostOp(info.opid, stream_info, info.settings));
    },
    true);

static popart::popx::OpxCreator<poptorch::HostOpx>
    host_opx_creator(poptorch_custom_ops::host_op);

static popart::RegisterShapeInferenceFunction host_op_shape_inference(
    poptorch_custom_ops::host_op, [](popart::ShapeInferenceContext &ctx) {
      // Get the stream info from the attribute map we passed to create the op.
      auto stream_info =
          poptorch::getMetadataFromAttributeMap(ctx.getAttributes());

      // Tell popart what the output should look like.
      for (std::uint32_t i = 0; i < stream_info->output_types.size(); ++i) {
        ctx.outInfo(i) = poptorch::shapeInferOutput(stream_info, i);
      }
    });
