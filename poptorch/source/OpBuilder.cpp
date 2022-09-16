// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/Utils.hpp"

#include "poptorch_logging/Logging.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
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

class SourceLocation {
public:
  // SourceLocation is considered enabled if a location or metadata
  // has been explicitly set.
  bool isEnabled() const { return _enabled; }

  void setLocation(const std::string &filename, std::uint64_t line) {
    _enabled = true;
    _dirty = true;
    _filename = filename;
    _line = line;
  }

  void setMetadata(const std::string &metadata) {
    _enabled = true;
    _dirty = true;
    _metadata = metadata;
  }

  const torch::jit::SourceRange &sourceRange() {
    if (_dirty) {
      _dirty = false;
      c10::optional<std::string> filename;
      if (!_filename.empty()) {
        filename = _filename;
      }
      ERROR_ON_MSG(
          _metadata.empty(),
          "[Internal] Metadata missing (setCurrentMetadata() missing)");
      auto source =
          std::make_shared<torch::jit::Source>(_metadata, filename, _line);
      _source_range = torch::jit::SourceRange(source, 0, 1);
    }
    return _source_range;
  }

private:
  bool _enabled{false};
  bool _dirty{false};
  torch::jit::SourceRange _source_range;
  std::string _metadata;
  std::string _filename;
  std::uint64_t _line;
} current_source_location = {};

} // namespace

void resetCurrentSourceLocation() {
  current_source_location = SourceLocation();
}

void setCurrentPythonCodeLocation(
    const torch::jit::SourceRange &source_location) {
  auto file_line_col = source_location.file_line_col();
  std::uint64_t line = 0;
  std::uint64_t col = 0;
  std::string filename;
  if (file_line_col) {
    std::tie(filename, line, col) = *file_line_col;
  }
  current_source_location.setLocation(filename, line);
}

void setCurrentMetadata(const std::string &metadata) {
  current_source_location.setMetadata(metadata);
}

WithNodeMetadata::WithNodeMetadata(torch::jit::Node *node) {
  // If no source location has been set yet
  // then the node won't contain any location information.
  if (current_source_location.isEnabled()) {
    std::string meta;
    auto sr = node->sourceRange();
    if (sr.source()) {
      meta = sr.source()->text();
    }
    setCurrentPythonCodeLocation(sr);
    setCurrentMetadata(meta);
  }
}

WithNodeMetadata::~WithNodeMetadata() {
  if (current_source_location.isEnabled()) {
    setCurrentPythonCodeLocation({});
    setCurrentMetadata("");
  }
}

torch::jit::Node *
createAndInsertNode(torch::jit::Graph *graph, torch::jit::NodeKind kind,
                    torch::jit::ArrayRef<torch::jit::Value *> inputs,
                    const ImplicitCast implicit_cast, OutputType output_type,
                    size_t num_outputs, c10::optional<at::ScalarType> dtype) {
  torch::jit::Node *new_node;

  if (implicit_cast != ImplicitCast::None && !inputs.empty()) {
    logging::LogContext ctx(std::string("implicitly casting inputs of ") +
                            kind.toQualString());
    auto possibly_cast_inputs = implicitCastInputs(&inputs, implicit_cast);
    ctx.clear();

    new_node = graph->create(kind, num_outputs);
    for (auto *input : possibly_cast_inputs) {
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
  insertNodeInGraph(graph, new_node);

  return new_node;
}

torch::jit::Value *insertConstant(torch::jit::Graph *graph,
                                  const torch::jit::IValue &val) {
  return graph->insertConstant(val, current_source_location.sourceRange());
}

void setSourceRangeToCurrentLocation(torch::jit::Node *node) {
  node->setSourceRange(current_source_location.sourceRange());
}

void insertNodeInGraph(torch::jit::Graph *graph, torch::jit::Node *new_node) {
  setSourceRangeToCurrentLocation(new_node);
  graph->insertNode(new_node);
  setAvailableMemoryAddPossibleInputOp(new_node);
}

void insertNodeBeforeNode(torch::jit::Node *new_node,
                          torch::jit::Node *insert_point) {
  setSourceRangeToCurrentLocation(new_node);
  new_node->insertBefore(insert_point);
  setAvailableMemoryAddPossibleInputOp(new_node);
}

void insertNodeAfterNode(torch::jit::Node *new_node,
                         torch::jit::Node *insert_point) {
  setSourceRangeToCurrentLocation(new_node);
  new_node->insertAfter(insert_point);
  setAvailableMemoryAddPossibleInputOp(new_node);
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
        resolved_output_type = onnxStrToScalarType(
            popart_compiler::onnxStrFromDtypeInt(onnx_dtype));
      } else {
        const auto &onnx_dtype = node->s(c10::attr::dtype);
        resolved_output_type = onnxStrToScalarType(onnx_dtype.c_str());
      }

      if (resolved_output_type == at::ScalarType::Float &&
          !isCompilingWithDispatcher()) {
        // Due to tracing not supporting Float16, the original type could be
        // either half or float 16.
        resolved_output_type = HALF_OR_FLOAT;
      }
    } else {
      // Without dtype, the input will be the correct type (or possibly
      // HALF_OR_FLOAT)
      resolved_output_type = scalarTypeFromInput(node, 0);
      // This may be needed in the lower to popart stage
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

  for (auto *output : node->outputs()) {
    output->setType(c10::TensorType::create(resolved_output_type, c10::nullopt,
                                            c10::nullopt, c10::nullopt));
  }
}

torch::jit::Node *tensorToConstant(torch::jit::Graph *graph,
                                   const at::Tensor &t,
                                   UseOfNode constant_use) {
  c10::Symbol symbol;
  switch (constant_use) {
  case UseOfNode::HostSideOnly:
    symbol = symbols::poptorch::host_side_tensor_constant;
    break;
  case UseOfNode::PopARTOnly:
    symbol = symbols::poptorch::tensor_constant;
    break;
  case UseOfNode::HostSideAndPopART:
    symbol = symbols::poptorch::host_and_ipu_side_tensor_constant;
    break;
  }

  torch::jit::Node *new_node = createAndInsertNode(graph, symbol);
  new_node->output()->inferTypeFrom(t);
  setNodeTensorAttrValue(new_node, t);

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

template <typename T, typename U>
torch::jit::Node *createConstant(torch::jit::Graph *graph,
                                 const std::vector<U> &data,
                                 const std::vector<int64_t> &new_shape,
                                 at::ScalarType scalar_type) {
  auto total_size = static_cast<size_t>(std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<int64_t>()));

  size_t stride = 0;
  if (data.size() != 1) {
    ERROR_ON(total_size != data.size());
    stride = 1;
  }
  auto t = at::empty(
      {new_shape},
      at::dtype(scalar_type).memory_format(c10::MemoryFormat::Contiguous));

  for (size_t i = 0; i < total_size; i++) {
    *(t.data_ptr<T>() + i) = static_cast<T>(data[i * stride]); // NOLINT
  }

  return tensorToConstant(graph, t);
}

torch::jit::Node *
createConstantInt(torch::jit::Graph *graph,
                  const std::vector<std::int64_t> &data,
                  const std::vector<std::int64_t> &new_shape) {
  return createConstant<std::int32_t>(graph, data, new_shape,
                                      at::ScalarType::Int);
}

torch::jit::Node *
createConstantLong(torch::jit::Graph *graph,
                   const std::vector<std::int64_t> &data,
                   const std::vector<std::int64_t> &new_shape) {
  return createConstant<std::int64_t>(graph, data, new_shape,
                                      at::ScalarType::Long);
}

torch::jit::Node *
createConstantFloat32(torch::jit::Graph *graph, const std::vector<double> &data,
                      const std::vector<std::int64_t> &new_shape) {
  return createConstant<float>(graph, data, new_shape, at::ScalarType::Float);
}

torch::jit::Node *
createConstantFloatLike(torch::jit::Graph *graph, torch::jit::Value *t,
                        const std::vector<double> &data,
                        const std::vector<std::int64_t> &new_shape) {
  at::ScalarType scalar_type =
      *t->type()->expect<c10::TensorType>()->scalarType();
  torch::jit::Node *new_node = createConstantFloat32(graph, data, new_shape);
  if (scalar_type == at::ScalarType::Half) {
    auto new_tensor = getNodeTensorAttrValue(new_node).to(scalar_type);
    setNodeTensorAttrValue(new_node, new_tensor);
    new_node->output()->inferTypeFrom(new_tensor);
  }
  return new_node;
}

torch::jit::Node *createInternalCast(torch::jit::Graph *graph,
                                     torch::jit::Value *A,
                                     const std::string &type) {
  // Convert from onnx string to a torch jit scalar object.
  c10::ScalarType as_type = onnxStrToScalarType(type.c_str());

  // Create the actual cast.
  return createCast(graph, A, as_type);
}

torch::jit::Node *createCast(torch::jit::Graph *graph, torch::jit::Value *A,
                             c10::ScalarType scalar_type) {
  std::string new_type = scalarTypeToOnnxString(scalar_type);

  auto *node = createCast(graph, {A}, new_type);

  const auto tensor_type = A->type()->expect<c10::TensorType>();
  node->output()->setType(tensor_type->withScalarType(scalar_type));
  return node;
}

static std::vector<std::int64_t>
convertPytorchPads(const std::vector<int64_t> &tensor_shape,
                   std::vector<int64_t> pad_shape) {
  // PopART requires padding for each dimension to be specified, so pad the
  // padding vector with zeros twice for each unspecified dim (one for
  // padding_before, one for padding_after)
  pad_shape.resize(tensor_shape.size() * 2, 0);

  // Converting from PyTorch to PopART requires two steps:
  // 1. Reverse the order
  // (beginN, endN, ..., begin1, end1) ->
  // (end1, begin1, ..., endN, beginN)
  std::reverse(pad_shape.begin(), pad_shape.end());
  // 2. Order padding dims by begin/end
  // (end1, begin1, ..., endN, beginN) ->
  // (begin1, ..., beginN, end1, ..., endN)
  //
  // This can be done with a single partition because begin and end
  // dims are at odd and even indices respectively. A stable partition
  // guarantees that the relative ordering of begin or end dims is unchanged
  std::stable_partition(pad_shape.begin(), pad_shape.end(),
                        [&](const int64_t &dim) {
                          auto index = &dim - std::addressof(pad_shape[0]);
                          return index % 2 == 1;
                        });

  return pad_shape;
}

torch::jit::Node *createConstantPad(torch::jit::Graph *graph,
                                    torch::jit::Value *A,
                                    const std::vector<int64_t> &pad_shape,
                                    float constant,
                                    bool direct_pad_shape_input) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::constant_pad, {A},
                          ImplicitCast::None, OutputType::AsFirstInput);
  new_node->is_(c10::Symbol::attr("pads"),
                direct_pad_shape_input
                    ? pad_shape
                    : convertPytorchPads(shapeFromTensor(A), pad_shape));
  new_node->f_(c10::Symbol::attr("value"), constant);
  return new_node;
}

torch::jit::Value *wrapInConstantVec(torch::jit::Graph *graph,
                                     const std::vector<int64_t> &data) {
  return createConstantInt(graph, data,
                           {static_cast<std::int64_t>(data.size())})
      ->output();
}

torch::jit::Node *createEdgePad(torch::jit::Graph *graph, torch::jit::Value *A,
                                const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::edge_pad, {A},
                          ImplicitCast::None, OutputType::AsFirstInput);
  new_node->is_(c10::Symbol::attr("pads"),
                convertPytorchPads(shapeFromTensor(A), pad_shape));
  return new_node;
}

torch::jit::Node *createReflectionPad(torch::jit::Graph *graph,
                                      torch::jit::Value *A,
                                      const std::vector<int64_t> &pad_shape) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::reflection_pad, {A},
                          ImplicitCast::None, OutputType::AsFirstInput);
  new_node->is_(c10::Symbol::attr("pads"),
                convertPytorchPads(shapeFromTensor(A), pad_shape));

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
                      std::int64_t domainVersion, std::int64_t numOutputs,
                      const std::string &attributes_id_str) {
  OutputType type =
      (numOutputs > 1) ? OutputType::Unknown : OutputType::AsFirstInput;

  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::custom_operation, inputs,
                          ImplicitCast::None, type, numOutputs);

  new_node->s_(c10::Symbol::attr("name"), name);
  new_node->s_(c10::Symbol::attr("domain"), domain);
  new_node->i_(c10::Symbol::attr("version"), domainVersion);
  new_node->i_(c10::Symbol::attr("num_outputs"), numOutputs);
  new_node->s_(c10::Symbol::attr("attributes_id"), attributes_id_str);

  return new_node;
}

torch::jit::Node *createAddUntypedInputTensor(torch::jit::Graph *graph,
                                              torch::jit::Value *input) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::add_untyped_input_tensor, {input});
  return new_node;
}

torch::jit::Node *createAddOutputTensor(torch::jit::Graph *graph,
                                        torch::jit::Value *output) {
  // We explicitly don't want to add this one as we want to add it based on the
  // position of the other node.
  torch::jit::Node *new_node =
      graph->create(symbols::poptorch::addOutputTensor, {output}, 0);
  return new_node;
}

torch::jit::Node *createStartForLoop(torch::jit::Graph *graph,
                                     torch::jit::Value *inputs) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::start_for_loop, inputs,
                          ImplicitCast::None, OutputType::Unknown, 0);
  return new_node;
}

torch::jit::Node *createEndForLoop(torch::jit::Graph *graph,
                                   torch::jit::Value *outputs,
                                   torch::jit::Value *inputs,
                                   std::int64_t trip_count) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::end_for_loop, {outputs, inputs});
  new_node->i_(c10::Symbol::attr("trip_count"), trip_count);
  const std::size_t num_outputs = outputs->node()->inputs().size();
  new_node->i_(c10::Symbol::attr("num_outputs"), num_outputs);
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
  if (possible_input != nullptr) {
    inputs.push_back(possible_input);
  }

  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::random_uniform, inputs, ImplicitCast::None,
      OutputType::AsDtypeOrAsPromoted, 1, dataType);

  new_node->is_(c10::attr::shape, shape);
  new_node->f_(c10::attr::high, high);
  new_node->f_(c10::attr::low, low);

  // At this point, the input is no longer needed
  if (possible_input != nullptr) {
    new_node->removeInput(0);
  }

  return new_node;
}

torch::jit::Node *createPrintIpuTensor(torch::jit::Graph *graph,
                                       torch::jit::Value *value,
                                       const std::string &title) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::ipu_print_tensor, {value});

  new_node->i_(c10::Symbol::attr("print_gradient"), 1);
  new_node->s_(c10::Symbol::attr("name"), "");
  new_node->s_(c10::Symbol::attr("title"), title);

  new_node->output()->setType(value->type());

  return new_node;
}

torch::jit::Node *createCallCpuOp(torch::jit::Graph *graph,
                                  const std::vector<torch::jit::Value *> &value,
                                  const std::string &id,
                                  torch::jit::Node *original_node) {
  const std::uint32_t num_outputs = original_node->outputs().size();
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::canonicalised_cpu_call, {value},
      ImplicitCast::None, OutputType::AsDtypeOrAsPromoted, num_outputs);

  new_node->s_(c10::Symbol::attr("ID"), id);

  for (std::uint32_t i = 0; i < num_outputs; ++i) {
    torch::jit::Value *old_out = original_node->output(i);
    torch::jit::Value *new_out = new_node->output(i);

    new_out->copyMetadata(old_out);
  }

  return new_node;
}

torch::jit::Node *createSetAvailableMemory(torch::jit::Graph *graph,
                                           torch::jit::Value *value,
                                           float proportion) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::set_available_memory, value);
  new_node->f_(c10::Symbol::attr("availableMemoryProportion"), proportion);

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

  new_node->s_(c10::Symbol::attr("mode"), mode);
  new_node->i_(c10::Symbol::attr("factor"), factor);
  new_node->i_(
      c10::Symbol::attr("keep_precision"),
      static_cast<torch::jit::IntAttr::ConstructorType>(keep_precision));

  new_node->output()->setType(matmul->type());

  return new_node;
}

torch::jit::Node *createBeginIpuBlock(torch::jit::Graph *graph,
                                      std::uint64_t stage_id,
                                      std::int64_t phase, std::int64_t ipu_id) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, c10::Symbol::fromQualString("poptorch::begin_ipu_block"), {},
      ImplicitCast::None, OutputType::Unknown, 0);
  new_node->i_(c10::Symbol::attr("stage"), stage_id);
  new_node->i_(c10::Symbol::attr("phase"), phase);
  new_node->i_(c10::Symbol::attr("ipu"), ipu_id);

  return new_node;
}

torch::jit::Node *
createOptimizerGroup(torch::jit::Graph *graph, std::uint64_t group,
                     const std::vector<torch::jit::Value *> &list_of_params) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::optimizer_group, list_of_params,
      ImplicitCast::None, OutputType::Unknown, 0);
  new_node->i_(c10::Symbol::attr("group"), group);

  return new_node;
}

torch::jit::Node *createRecomputationCheckpoint(torch::jit::Graph *graph,
                                                torch::jit::Value *value) {
  return createAndInsertNode(graph, symbols::poptorch::recomputation_checkpoint,
                             {value}, ImplicitCast::None,
                             OutputType::AsFirstInput);
}

torch::jit::Node *createUnfold(torch::jit::Graph *graph,
                               torch::jit::Value *value, int64_t dimension,
                               int64_t size, int64_t step) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::unfold, {value},
                          ImplicitCast::None, OutputType::AsFirstInput);
  new_node->i_(c10::Symbol::fromQualString("attr::dimension"), dimension);
  new_node->i_(c10::Symbol::fromQualString("attr::size"), size);
  new_node->i_(c10::Symbol::fromQualString("attr::step"), step);

  return new_node;
}

torch::jit::Node *createMultiConvPart(torch::jit::Graph *graph,
                                      torch::jit::Node *conv_node) {
  ERROR_ON_MSG(conv_node->kind() != symbols::popart::conv,
               "Can only create multi_conv_part from conv node");

  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::multi_conv_part, conv_node->inputs(),
      ImplicitCast::All, OutputType::AsImplicitCastPromoted);

  new_node = new_node->copyAttributes(*conv_node);
  new_node->output()->setType(conv_node->output()->type());
  return new_node;
}

torch::jit::Node *createGru(torch::jit::Graph *graph,
                            const std::vector<torch::jit::Value *> &args,
                            int64_t hidden_size) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::gru, args, ImplicitCast::All,
      OutputType::AsImplicitCastPromoted, 2);
  new_node->i_(c10::attr::hidden_size, hidden_size);

  return new_node;
}

torch::jit::Node *createRnn(torch::jit::Graph *graph,
                            const std::vector<torch::jit::Value *> &args,
                            const std::vector<std::string> &activations) {
  torch::jit::Node *new_node = createAndInsertNode(
      graph, symbols::poptorch::rnn, args, ImplicitCast::All,
      OutputType::AsImplicitCastPromoted, 2);
  new_node->ss_(c10::Symbol::attr("activations"), activations);
  return new_node;
}

torch::jit::Node *createPrelu(torch::jit::Graph *graph, torch::jit::Value *self,
                              torch::jit::Value *weight) {
  torch::jit::Node *new_node =
      createAndInsertNode(graph, symbols::poptorch::prelu, {self, weight},
                          ImplicitCast::None, OutputType::AsFirstInput);

  return new_node;
}

/*
 * Auto generated operation.
 */
#include "CompilerOps.cpp.inc"

} // namespace poptorch
