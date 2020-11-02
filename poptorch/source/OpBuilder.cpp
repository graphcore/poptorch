// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popart_compiler/PopartEnums.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

#include "PoptorchSymbols.hpp"

namespace poptorch {
namespace {

at::ScalarType scalarTypeFromInput(const torch::jit::Node *node, size_t num) {
  ERROR_ON_MSG(node->inputs().size() <= num,
               "Cannot get scalar type from input " << num
                                                    << " as it does not exist");
  return *node->input(num)->type()->expect<c10::TensorType>()->scalarType();
}
} // namespace

torch::jit::Node *
createAndInsertNode(torch::jit::Graph *graph, torch::jit::NodeKind kind,
                    torch::jit::ArrayRef<torch::jit::Value *> inputs,
                    const ImplicitCast implicit_cast, OutputType output_type,
                    size_t num_outputs, c10::optional<at::ScalarType> dtype) {
  torch::jit::Node *new_node;

  if (implicit_cast != ImplicitCast::None && !inputs.empty()) {
    logging::LogContext ctx(std::string("implicitly casting inputs of ") +
                            kind.toQualString());
    auto possibly_casted_inputs = implicitCastInputs(&inputs, implicit_cast);
    ctx.clear();

    new_node = graph->create(kind, num_outputs);
    for (auto input : possibly_casted_inputs) {
      new_node->addInput(input);
    }
  } else {
    new_node = graph->create(kind, inputs, num_outputs);
  }

  if (dtype) {
    if (*dtype != at::ScalarType::Undefined) {
      new_node->s_(c10::attr::dtype, scalarTypeToOnnxString(*dtype));
    }
  }

  setNodeOutputsTypes(new_node, implicit_cast, output_type);
  graph->insertNode(new_node);
  return new_node;
}

// Sets the scalar types of every output of a node
void setNodeOutputsTypes(torch::jit::Node *node,
                         const ImplicitCast implicit_cast,
                         const OutputType output_type) {
  at::ScalarType resolved_output_type;

  switch (output_type) {
  case OutputType::Unknown: {
    return;
  }
  case OutputType::AsFirstInput: {
    resolved_output_type = scalarTypeFromInput(node, 0);
    break;
  }
  case OutputType::FirstAsFirstInputSecondAlwaysInt: {
    node->output(0)->setType(
        c10::TensorType::create(scalarTypeFromInput(node, 0), c10::nullopt,
                                c10::nullopt, c10::nullopt));
    node->output(1)->setType(c10::TensorType::create(
        at::ScalarType::Int, c10::nullopt, c10::nullopt, c10::nullopt));
    return;
  }
  case OutputType::AsThirdInput: {
    resolved_output_type = scalarTypeFromInput(node, 2);
    break;
  }
  case OutputType::AsImplicitCastPromoted: {
    size_t input_idx = (implicit_cast == ImplicitCast::ExceptFirst) ? 1 : 0;
    resolved_output_type = scalarTypeFromInput(node, input_idx);
    break;
  }
  case OutputType::AsDtype:
    [[fallthrough]];
  case OutputType::AsDtypeOrAsPromoted: {
    // Cast uses "to" not "dtype" and a string
    if (node->kind() == symbols::popart::cast) {
      // Type is handled in OpBuilder.cpp
      return;
    }

    if (node->hasAttribute(c10::attr::dtype)) {
      if (node->kindOf(c10::attr::dtype) == torch::jit::AttributeKind::i) {
        const auto onnx_dtype = node->i(c10::attr::dtype);
        resolved_output_type =
            onnxStrToScalarType(onnxStrFromDtypeInt(onnx_dtype));
      } else {
        const auto &onnx_dtype = node->s(c10::attr::dtype);
        resolved_output_type = onnxStrToScalarType(onnx_dtype.c_str());
      }

      if (resolved_output_type == at::ScalarType::Float) {
        // Due to tracing not supporting Float16, the original type could be
        // either half or float 16.
        resolved_output_type = HALF_OR_FLOAT;
      }
    } else {
      // Without dtype, the input will be the correct type (or possibly
      // HALF_OR_FLOAT)
      resolved_output_type = scalarTypeFromInput(node, 0);
      // This may be needed in the lower to popart stage.
      node->s_(c10::attr::dtype, scalarTypeToOnnxString(resolved_output_type));
    }
    break;
  }
  case OutputType::AlwaysBool: {
    resolved_output_type = at::ScalarType::Bool;
    break;
  }
  case OutputType::AlwaysFloat: {
    resolved_output_type = at::ScalarType::Float;
    break;
  }
  case OutputType::AlwaysInt: {
    resolved_output_type = at::ScalarType::Int;
    break;
  }
  case OutputType::AlwaysUint8: {
    resolved_output_type = at::ScalarType::Byte;
    break;
  }
  default: {
    ERROR("Unsupported output_type in setNodeOutputsTypes");
  }
  }

  for (auto output : node->outputs()) {
    output->setType(c10::TensorType::create(resolved_output_type, c10::nullopt,
                                            c10::nullopt, c10::nullopt));
  }
}

torch::jit::Node *tensorToConstant(torch::jit::Graph *graph,
                                   const at::Tensor &t) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::tensor_constant);
  new_node->output()->inferTypeFrom(t);
  new_node->t_(c10::attr::value, t);

  return new_node;
}

/*
 * Manually added operation.
 */
torch::jit::Node *createReshape(torch::jit::Graph *graph, torch::jit::Value *A,
                                const std::vector<int64_t> &new_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::popart::reshape_static_shape, {A});
  new_node->is_(c10::attr::shape, new_shape);
  new_node->output()->setType(
      A->type()->expect<c10::TensorType>()->withSizes(new_shape));
  return new_node;
}

torch::jit::Node *createConstantInt(torch::jit::Graph *graph,
                                    const std::vector<int64_t> &data,
                                    const std::vector<int64_t> &new_shape) {
  auto total_size = static_cast<size_t>(std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<int64_t>()));

  size_t stride = 0;
  if (data.size() != 1) {
    ERROR_ON(total_size != data.size());
    stride = 1;
  }

  auto t =
      at::empty({new_shape}, at::dtype(at::ScalarType::Int)
                                 .memory_format(c10::MemoryFormat::Contiguous));

  for (size_t i = 0; i < total_size; i++) {
    // NOLINTNEXTLINE
    *(t.data_ptr<int32_t>() + i) = static_cast<int32_t>(data[i * stride]);
  }

  // Use bounds checking for the last element
  data.at((total_size - 1) * stride);

  return tensorToConstant(graph, t);
}

torch::jit::Node *createConstantFloat(torch::jit::Graph *graph,
                                      const std::vector<double> &data,
                                      const std::vector<int64_t> &new_shape) {
  auto total_size = static_cast<size_t>(std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>()));
  ERROR_ON(total_size == 0);

  size_t stride = 0;

  if (data.size() != 1) {
    ERROR_ON(total_size != data.size());
    stride = 1;
  }

  auto t =
      at::empty({new_shape}, at::dtype(at::ScalarType::Float)
                                 .memory_format(c10::MemoryFormat::Contiguous));

  for (size_t i = 0; i < total_size; i++) {
    // NOLINTNEXTLINE
    *(t.data_ptr<float>() + i) = static_cast<float>(data[i * stride]);
  }

  // Use bounds checking for the last element
  data.at((total_size - 1) * stride);

  return tensorToConstant(graph, t);
}

torch::jit::Node *createConstantFloat16(torch::jit::Graph *graph,
                                        const std::vector<double> &data,
                                        const std::vector<int64_t> &new_shape) {
  torch::jit::Node *new_node = createConstantFloat(graph, data, new_shape);
  new_node->t_(c10::attr::value,
               new_node->t(c10::attr::value).to(at::ScalarType::Half));

  return new_node;
}

torch::jit::Node *createCast(torch::jit::Graph *graph, torch::jit::Value *A,
                             c10::ScalarType scalar_type) {
  std::string new_type = scalarTypeToOnnxString(scalar_type);
  auto node = createCast(graph, {A}, new_type);

  const auto tensor_type = A->type()->expect<c10::TensorType>();
  node->output()->setType(tensor_type->withScalarType(scalar_type));
  return node;
}

static std::vector<std::int64_t>
convertPytorchPads(const std::vector<int64_t> &pad_shape) {
  std::vector<int64_t> tmp = pad_shape;

  // Work out how many dimensions we are padding. Each dimension is in the form
  // (begin1, end1, beginN, endN)
  const std::size_t num_dimensions = tmp.size() / 2;

  ERROR_ON_MSG(num_dimensions > 3, "Internal error: Unsupported number of "
                                   "dimensions in constant pad operation.");

  // Move from pytorch (begin1, end1, beginN, endN) to popart (begin1, beginN,
  // end1, endN) It's also in reverse order.
  if (num_dimensions == 2) {
    // Move to popart ordering.
    std::swap(tmp[1], tmp[2]);

    // Reverse the order.
    std::swap(tmp[0], tmp[1]);
    std::swap(tmp[2], tmp[3]);

  } else if (num_dimensions == 3) {
    // Move to popart ordering and reverse.
    tmp = {
        // The begins.
        tmp[4],
        tmp[2],
        tmp[0],

        // The ends.
        tmp[5],
        tmp[3],
        tmp[1],
    };
  }
  // Padding is applying to the * dimensions in (N, C, *) but popart allows N/C
  // to be specified as well. So add 4 zeros to make it compatable. (N_beg,
  // C_beg, *_beg, N_end, C_end, *_end).
  tmp.insert(tmp.begin(), 0);
  tmp.insert(tmp.begin(), 0);

  // Insert after the first two zeros and after however many dimensions there
  // are.
  tmp.insert(tmp.begin() + num_dimensions + 2, 0);
  tmp.insert(tmp.begin() + num_dimensions + 2, 0);

  return tmp;
}

torch::jit::Node *createConstantPad(torch::jit::Graph *graph,
                                    torch::jit::Value *A,
                                    const std::vector<int64_t> &pad_shape,
                                    float constant) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::constant_pad, {A},
                          ImplicitCast::None, OutputType::AsFirstInput);
  new_node->is_(c10::Symbol::fromQualString("attr::pads"),
                convertPytorchPads(pad_shape));
  new_node->f_(c10::Symbol::fromQualString("attr::value"), constant);
  return new_node;
}

torch::jit::Node *createEdgePad(torch::jit::Graph *graph, torch::jit::Value *A,
                                const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::edge_pad, {A},
                          ImplicitCast::None, OutputType::AsFirstInput);
  new_node->is_(c10::Symbol::fromQualString("attr::pads"),
                convertPytorchPads(pad_shape));
  return new_node;
}

torch::jit::Node *createReflectionPad(torch::jit::Graph *graph,
                                      torch::jit::Value *A,
                                      const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::reflection_pad, {A},
                          ImplicitCast::None, OutputType::AsFirstInput);
  new_node->is_(c10::Symbol::fromQualString("attr::pads"),
                convertPytorchPads(pad_shape));

  return new_node;
}

torch::jit::Node *createAddNotInPlace(torch::jit::Graph *graph,
                                      torch::jit::Value *A,
                                      torch::jit::Value *B) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::add_not_in_place, {A, B}, ImplicitCast::All,
      OutputType::AsImplicitCastPromoted);
  return new_node;
}

torch::jit::Node *
createCustomOperation(torch::jit::Graph *graph,
                      const std::vector<torch::jit::Value *> &inputs,
                      const std::string &name, const std::string &domain,
                      std::int64_t domainVersion, std::int64_t numOutputs) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::custom_operation, inputs,
                          ImplicitCast::None, OutputType::Unknown, numOutputs);

  new_node->s_(c10::Symbol::fromQualString("attr::name"), name);
  new_node->s_(c10::Symbol::fromQualString("attr::domain"), domain);
  new_node->i_(c10::Symbol::fromQualString("attr::version"), domainVersion);
  new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"), numOutputs);

  return new_node;
}

torch::jit::Node *
createRandomNormal(torch::jit::Graph *graph,
                   const std::vector<torch::jit::Value *> &possible_inputs,
                   const std::vector<int64_t> &shape, float mean, float scale,
                   at::ScalarType dataType) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::random_normal, possible_inputs,
      ImplicitCast::All, OutputType::AsDtypeOrAsPromoted, 1, dataType);

  new_node->is_(c10::attr::shape, shape);
  new_node->f_(c10::attr::mean, mean);
  new_node->f_(c10::attr::scale, scale);

  // At this point, the input is no longer needed
  for (size_t i = 0; i < possible_inputs.size(); i++) {
    new_node->removeInput(0); // input 1 and input 0
  }

  return new_node;
}

torch::jit::Node *createRandomUniform(torch::jit::Graph *graph,
                                      torch::jit::Value *possible_input,
                                      const std::vector<int64_t> &shape,
                                      float high, float low,
                                      at::ScalarType dataType) {
  std::vector<torch::jit::Value *> inputs;
  if (possible_input) {
    inputs.push_back(possible_input);
  }

  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::random_uniform, inputs, ImplicitCast::None,
      OutputType::AsDtypeOrAsPromoted, 1, dataType);

  new_node->is_(c10::attr::shape, shape);
  new_node->f_(c10::attr::high, high);
  new_node->f_(c10::attr::low, low);

  // At this point, the input is no longer needed
  if (possible_input) {
    new_node->removeInput(0);
  }

  return new_node;
}

torch::jit::Node *createPrintIpuTensor(torch::jit::Graph *graph,
                                       torch::jit::Value *value) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::ipu_print_tensor, {value},
                          ImplicitCast::None, OutputType::AsFirstInput);
  return new_node;
}

torch::jit::Node *createSetAvailableMemory(torch::jit::Graph *graph,
                                           torch::jit::Value *value,
                                           float proportion) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::set_available_memory, {value});
  new_node->f_(c10::Symbol::fromQualString("attr::availableMemoryProportion"),
               proportion);

  new_node->output()->setType(value->type());

  return new_node;
}

torch::jit::Node *createSetMatMulSerialization(torch::jit::Graph *graph,
                                               torch::jit::Value *matmul,
                                               const std::string &mode,
                                               int64_t factor,
                                               bool keep_precision) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::set_matmul_serialization, {matmul});

  new_node->s_(c10::Symbol::fromQualString("attr::mode"), mode);
  new_node->i_(c10::Symbol::fromQualString("attr::factor"), factor);
  new_node->i_(c10::Symbol::fromQualString("attr::keep_precision"),
               keep_precision);

  new_node->output()->setType(matmul->type());

  return new_node;
}

torch::jit::Node *createBeginIpuBlock(torch::jit::Graph *graph,
                                      std::uint64_t stage_id,
                                      std::int64_t phase, std::int64_t ipu_id) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, c10::Symbol::fromQualString("poptorch::begin_ipu_block"), {},
      ImplicitCast::None, OutputType::Unknown, 0);
  new_node->i_(c10::Symbol::fromQualString("attr::stage"), stage_id);
  new_node->i_(c10::Symbol::fromQualString("attr::phase"), phase);
  new_node->i_(c10::Symbol::fromQualString("attr::ipu"), ipu_id);

  return new_node;
}

torch::jit::Node *
createOptimizerGroup(torch::jit::Graph *graph, std::uint64_t group,
                     const std::vector<torch::jit::Value *> &list_of_params) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::optimizer_group, list_of_params,
      ImplicitCast::None, OutputType::Unknown, 0);
  new_node->i_(c10::Symbol::fromQualString("attr::group"), group);

  return new_node;
}

/*
 * Auto generated operation.
 */
#include "CompilerOps.cpp.inc"

} // namespace poptorch
