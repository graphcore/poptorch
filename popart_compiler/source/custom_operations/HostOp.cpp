// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/iarray.hpp>

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
#include <popart/operators.hpp>

#include <CustomOps.hpp>

#include "popart_compiler/CompilerImpl.hpp"

namespace poptorch_custom_ops {

const char host_op_metadata_attr[] = "func_info";

} // namespace poptorch_custom_ops

/*
 * A popart custom operation to handle Host operations. Takes a callback and
 * sets up the IPU->CPU communication for the needed tensors.
 */
namespace poptorch {

namespace {

// Get the popart type info for a given output from the stream metadata.
popart::TensorInfo shapeInferOutput(detail::CallbackInternalMetadata *func_info,
                                    std::uint32_t i) {
  // Get type and shape from metadata.
  const popart::DataType type =
      poptorch::popartTypeFromPoptorch(func_info->output_types[i]);
  const std::vector<std::size_t> &shape = func_info->output_shapes[i];

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
  std::int64_t as_int = attrs.getAttribute<popart::Attributes::Int>(
      poptorch_custom_ops::host_op_metadata_attr);

  logging::trace("Pointer retrieved by CPU op {}", as_int);

  std::intptr_t as_ptr = static_cast<std::intptr_t>(as_int);

  logging::trace("Casted from {} to {}", as_int, as_ptr);

  // Cast to the correct type.
  // NOLINTNEXTLINE performance-no-int-to-ptr
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
      : popart::Op(_opid, settings_), func_info(info) {}

  // Configure the output popart Tensor
  void setup() override {
    // Tell popart what the output should look like.
    for (std::uint32_t i = 0; i < func_info->output_types.size(); ++i) {
      outInfo(i) = shapeInferOutput(func_info, i);
    }
  }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<HostOp>(*this);
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  poptorch::detail::CallbackInternalMetadata *func_info;
};

class HostOpx : public popart::popx::Opx {
public:
  HostOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<HostOp>(op, poptorch_custom_ops::host_op);

    func_info = dynamic_cast<HostOp *>(op)->func_info;
  }

  void grow(poplar::program::Sequence &sequence) const override {
    poplar::Graph &graph = this->graph();

    // Get basic op info from metadata.
    const std::uint32_t num_inputs = func_info->input_types.size();
    const std::uint32_t num_outputs = func_info->output_types.size();

    // For each input create the FIFO and copy from it into the poplar tensor
    // popart has already created/
    std::vector<poplar::Graph::HostFunctionArgument> input_args;
    std::vector<poplar::Tensor> inputs;
    inputs.reserve(num_inputs);
    input_args.reserve(num_inputs);
    for (std::uint32_t input_index = 0; input_index < num_inputs;
         ++input_index) {
      // poplar::Tensor from popart.
      poplar::Tensor input_tensor = getInTensor(input_index);
      inputs.push_back(input_tensor);
      input_args.emplace_back(input_tensor.elementType(),
                              input_tensor.numElements());
    }

    std::vector<poplar::Graph::HostFunctionArgument> output_args;
    std::vector<poplar::Tensor> outputs;
    outputs.reserve(num_outputs);
    output_args.reserve(num_outputs);
    for (std::uint32_t output = 0; output < num_outputs; ++output) {
      const poplar::Type type =
          poptorch::poplarTypeFromPoptorch(func_info->output_types[output]);

      const std::vector<std::size_t> &shape = func_info->output_shapes[output];

      // Add the poplar tensor.
      std::string name = func_info->handle + "::out" + std::to_string(output);
      poplar::Tensor output_tensor = graph.addVariable(
          type, shape, poplar::VariableMappingMethod::LINEAR, std::move(name));

      outputs.push_back(output_tensor);
      output_args.emplace_back(output_tensor.elementType(),
                               output_tensor.numElements());

      // Tell popart this is the output.
      setOutTensor(output, output_tensor);
    }

    poplar::HostFunction hf =
        graph.addHostFunction(func_info->handle, input_args, output_args);
    sequence.add(poplar::program::Call(hf, inputs, outputs));
  }

  poptorch::detail::CallbackInternalMetadata *func_info;
};

} // namespace poptorch

static popart::OpCreator<poptorch::HostOp> host_op_creator(
    {{poptorch_custom_ops::host_op, {}}},
    [](const popart::OpCreatorInfo &info) {
      // Get the stream info from the attribute map we passed to
      // create the op.
      auto *func_info = poptorch::getMetadataFromAttributeMap(info.attributes);

      return std::unique_ptr<popart::Op>(
          new poptorch::HostOp(info.opid, func_info, info.settings));
    },
    true);

static popart::popx::OpxCreator<poptorch::HostOpx>
    host_opx_creator(poptorch_custom_ops::host_op);

static popart::RegisterShapeInferenceFunction host_op_shape_inference(
    poptorch_custom_ops::host_op, [](popart::ShapeInferenceContext &ctx) {
      // Get the stream info from the attribute map we passed to create the op.
      auto *func_info =
          poptorch::getMetadataFromAttributeMap(ctx.getAttributes());

      // Tell popart what the output should look like.
      for (std::uint32_t i = 0; i < func_info->output_types.size(); ++i) {
        ctx.outInfo(i) = poptorch::shapeInferOutput(func_info, i);
      }
    });
