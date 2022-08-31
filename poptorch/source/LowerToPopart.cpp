// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/LowerToPopart.hpp"

#include <experimental/filesystem>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <list>
#include <random>
#include <utility>

#include "PoptorchSymbols.hpp"
#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/PopartEnums.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/InplaceOps.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace fs = std::experimental::filesystem;

namespace poptorch {

namespace {

std::string getModelProtoFilename() {
  if (const char *proto_file = std::getenv("POPTORCH_EXPORT_PROTO_FILE")) {
    fs::path file = fs::absolute(proto_file);
    fs::path dir = file;
    if (dir.has_extension()) {
      dir.remove_filename();
    } else {
      file += "/model.proto";
    }
    fs::create_directories(dir);
    logging::info(
        "POPTORCH_EXPORT_PROTO_FILE set: saving model prototype to {}", file);
    return file;
  }
  return "";
}

// Mapping between the SSA values of torch jit with the ssa values of popart.
// Each Value is either a single tensor, tuple or list (Note: nested tuples are
// stored flattened).
class ValueMap {
public:
  using TensorList = std::vector<popart_compiler::TensorId>;

  popart_compiler::TensorId tensor(torch::jit::Value *value) const;
  const TensorList &listTuple(torch::jit::Value *value) const;

  // Return the list of tensors without checking if it's a tuple, list or a
  // single tensor.
  const TensorList &tensors(torch::jit::Value *value) const;

  bool hasTensor(torch::jit::Value *value) const {
    return _map.count(value) == 1;
  }

  void setTensor(torch::jit::Value *value, popart_compiler::TensorId id);
  void setList(torch::jit::Value *value, const TensorList &tensors);
  void setTuple(torch::jit::Value *value, const TensorList &tensors);

private:
  struct Data {
    explicit Data(popart_compiler::TensorId id)
        : type(popart_compiler::OutputElemType::Tensor) {
      tensors.push_back(id);
    }

    Data(TensorList tuple, popart_compiler::OutputElemType type_)
        : type(type_), tensors(std::move(tuple)) {}
    popart_compiler::OutputElemType type;
    TensorList tensors;
  };
  std::unordered_map<torch::jit::Value *, Data> _map;
};

popart_compiler::TensorId ValueMap::tensor(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  ERROR_ON_MSG(it->second.type != popart_compiler::OutputElemType::Tensor,
               value->debugName() << " is not a tensor");
  ERROR_ON(it->second.tensors.size() != 1);
  return it->second.tensors.front();
}

const ValueMap::TensorList &
ValueMap::listTuple(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  ERROR_ON_MSG((it->second.type != popart_compiler::OutputElemType::Tuple &&
                it->second.type != popart_compiler::OutputElemType::List),
               value->debugName() << " is not a tuple or list");
  return it->second.tensors;
}

const ValueMap::TensorList &ValueMap::tensors(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  return it->second.tensors;
}

void ValueMap::setTensor(torch::jit::Value *value,
                         popart_compiler::TensorId id) {
  ERROR_ON_MSG(!_map.emplace(value, Data(id)).second,
               "Value " << value->debugName() << " already present in the map");
}

void ValueMap::setList(torch::jit::Value *value,
                       const ValueMap::TensorList &tensors) {
  ERROR_ON_MSG(
      !_map.emplace(value, Data(tensors, popart_compiler::OutputElemType::List))
           .second,
      "Value " << value->debugName() << " already present in the map");
}

void ValueMap::setTuple(torch::jit::Value *value,
                        const ValueMap::TensorList &tensors) {
  ERROR_ON_MSG(
      !_map.emplace(value,
                    Data(tensors, popart_compiler::OutputElemType::Tuple))
           .second,
      "Value " << value->debugName() << " already present in the map");
}

/*
 * Static helper functions.
 */

std::string typeToPopartStr(at::ScalarType type) {
  if (type == at::ScalarType::Float || type == at::ScalarType::Double) {
    return "FLOAT";
  }
  if (type == at::ScalarType::Half) {
    return "FLOAT16";
  }
  if (type == at::ScalarType::Short) {
    return "INT16";
  }
  if (type == at::ScalarType::Int || type == at::ScalarType::Long) {
    return "INT32";
  }
  if (type == at::ScalarType::Bool) {
    return "BOOL";
  }
  if (type == at::ScalarType::Char) {
    return "INT8";
  }
  if (type == at::ScalarType::Byte) {
    return "UINT8";
  }

  logging::err("Unimplemented type '{}'", type);
  return "UNIMPLEMENTED";
}

std::vector<int64_t> getTensorDimensions(const at::Tensor &tensor) {
  std::vector<int64_t> dims;
  std::transform(tensor.sizes().begin(), tensor.sizes().end(),
                 std::back_inserter(dims), [](std::int64_t i) { return i; });
  return dims;
}

at::ScalarType fromPopartType(const popart_compiler::PopartType type) {
  switch (type) {
  case popart_compiler::PopartType::UINT8: {
    return at::ScalarType::Byte;
  }
  case popart_compiler::PopartType::INT8: {
    return at::ScalarType::Char;
  }
  case popart_compiler::PopartType::INT16:
  case popart_compiler::PopartType::UINT16: {
    return at::ScalarType::Short;
  }
  case popart_compiler::PopartType::INT32:
  case popart_compiler::PopartType::UINT32: {
    return at::ScalarType::Int;
  }
  case popart_compiler::PopartType::INT64: {
    return at::ScalarType::Long;
  }
  case popart_compiler::PopartType::BOOL: {
    return at::ScalarType::Bool;
  }
  case popart_compiler::PopartType::FLOAT: {
    return at::ScalarType::Float;
  }
  case popart_compiler::PopartType::FLOAT16: {
    return at::ScalarType::Half;
  }
  case popart_compiler::PopartType::BFLOAT16: {
    return at::ScalarType::BFloat16;
  }
  case popart_compiler::PopartType::DOUBLE: {
    return at::ScalarType::Double;
  }
  case popart_compiler::PopartType::COMPLEX64: {
    return at::ScalarType::ComplexFloat;
  }
  case popart_compiler::PopartType::COMPLEX128: {
    return at::ScalarType::ComplexDouble;
  }
  default:
    ERROR("Unsupported PopART data type " << toPopartTypeStr(type));
  }
}

popart_compiler::PopartType toPopartType(const at::ScalarType type) {
  switch (type) {
  case at::ScalarType::Byte: {
    return popart_compiler::PopartType::UINT8;
  }
  case at::ScalarType::Char: {
    return popart_compiler::PopartType::INT8;
  }
  case at::ScalarType::Short: {
    return popart_compiler::PopartType::INT16;
  }
  case at::ScalarType::Int: {
    return popart_compiler::PopartType::INT32;
  }
  case at::ScalarType::Long: {
    return popart_compiler::PopartType::INT64;
  }
  case at::ScalarType::Bool: {
    return popart_compiler::PopartType::BOOL;
  }
  case at::ScalarType::Float: {
    return popart_compiler::PopartType::FLOAT;
  }
  case at::ScalarType::Half: {
    return popart_compiler::PopartType::FLOAT16;
  }
  case at::ScalarType::BFloat16: {
    return popart_compiler::PopartType::BFLOAT16;
  }
  case at::ScalarType::Double: {
    return popart_compiler::PopartType::DOUBLE;
  }
  case at::ScalarType::ComplexFloat: {
    return popart_compiler::PopartType::COMPLEX64;
  }
  case at::ScalarType::ComplexDouble: {
    return popart_compiler::PopartType::COMPLEX128;
  }
  default:
    ERROR("Unsupported PyTorch scalar type " << toString(type));
  }
}

void platformAgnosticTypeInfoFromIRType(
    torch::jit::Value *value, std::vector<popart_compiler::PopartType> *types,
    std::vector<std::vector<std::size_t>> *shapes) {
  std::shared_ptr<c10::TensorType> tensor_type =
      value->type()->expect<c10::TensorType>();
  c10::ScalarType as_scalar = *tensor_type->scalarType();

  types->push_back(toPopartType(as_scalar));

  c10::VaryingShape shape = tensor_type->sizes();

  shapes->emplace_back();

  for (std::uint32_t i = 0; i < *shape.size(); ++i) {
    shapes->back().push_back(*shape[i]);
  }
}

} // namespace

namespace detail {
/*
 * Implementation of the lowering operation.
 */
class LowerToPopartImpl {
public:
  LowerToPopartImpl(torch::jit::Graph *g, InplaceGraphInfo &&inplace_info,
                    bool training,
                    std::vector<popart_compiler::Optimizer> &&opt,
                    const popart_compiler::SessionOptions &options,
                    const AttributeAccessor &attribute_accessor,
                    CPUCallbackMap &&callback, const AnchorList &&anchors);
  void setParameters(const std::vector<at::Tensor> &params,
                     const std::vector<std::string> &parameter_names);

  void lower(std::vector<at::Tensor> *in_tensors);

  std::shared_ptr<PoplarExecutable> compile();
  std::shared_ptr<PoplarExecutable>
  loadExecutableFromFile(const std::string &input_filename);

private:
  void printWasLoweredDebug(const torch::jit::Node *node,
                            popart_compiler::TensorId first_output_tensor);
  torch::jit::Graph &_graph;

  bool _lowered;
  bool _built_in_params;

  std::vector<at::Tensor> _parameters;
  std::vector<std::string> _parameter_names;
  InplaceGraphInfo _inplace_info;

  std::vector<popart_compiler::TensorId> _input_tensor_hooks;

  std::vector<popart_compiler::TensorId> _output_tensor_hooks;

  ValueMap _value_map;

  // Optimizer from the user.
  const std::vector<popart_compiler::Optimizer> _optimizers;

  // Tensors to be anchored other than outputs
  const AnchorList &_anchors;

  using FunctionType = std::function<popart_compiler::TensorId(
      const std::vector<popart_compiler::TensorId> &inputs,
      torch::jit::Node *)>;
  std::unordered_map<c10::Symbol, FunctionType> _functionToImplementation;

  popart_compiler::Compiler _compiler;

  CPUCallbackMap _callbacks;

  void lowerParameters(std::vector<at::Tensor> *in_tensors);

  void lowerBody();

  void lowerReturn();

  std::string tensorNames(std::int64_t first_tensor, std::int64_t num_tensors);
  std::string tensorNames(const ValueMap::TensorList &tensors);

  std::string tensorTypesAndShapes(std::int64_t first_tensor,
                                   std::int64_t num_tensors);
  std::string tensorTypesAndShapes(const ValueMap::TensorList &tensors);

  void validateOutputShapeAndType(popart_compiler::TensorId output_tensor,
                                  torch::jit::Node *node,
                                  std::uint64_t node_output);
};

namespace {
// Remove from vec all elements vec[i] for which mask[i] is false
template <typename T>
void maskVector(std::vector<T> *vec, const std::vector<bool> &mask,
                size_t ignore_first = 0) {
  auto predicate = [&mask, &vec, ignore_first](const T &val) {
    auto idx = static_cast<size_t>(&val - &(*vec->begin()));
    if (idx < ignore_first) {
      return false;
    }
    return !mask.at(idx - ignore_first);
  };

  auto erase_begin = std::remove_if(vec->begin(), vec->end(), predicate);
  vec->erase(erase_begin, vec->end());
}
} // namespace

void LowerToPopartImpl::setParameters(
    const std::vector<at::Tensor> &params,
    const std::vector<std::string> &parameter_names) {
  _parameters = params;
  _parameter_names = parameter_names;
  _built_in_params = false;
}

/*
 * Lower to popart impl.
 */
std::shared_ptr<PoplarExecutable> LowerToPopartImpl::compile() {
  ERROR_ON_MSG(!_lowered, "You need to lower() the graph first");

  logging::LogContext ctx("LowerToPopart::compile");
  // Init the session, this also involves compiling to poplar.
  _compiler.initSession(_optimizers, getModelProtoFilename().c_str());

  _compiler.compileAndPrepareDevice();

  std::vector<at::ScalarType> data_types;
  for (auto id : _output_tensor_hooks) {
    data_types.emplace_back(fromPopartType(_compiler.getPopartType(id)));
  }

  return std::make_shared<PoplarExecutable>(
      std::move(_compiler), std::move(_input_tensor_hooks),
      std::move(_output_tensor_hooks), std::move(data_types), _parameter_names,
      std::move(_inplace_info));
}

std::shared_ptr<PoplarExecutable>
LowerToPopartImpl::loadExecutableFromFile(const std::string &input_filename) {
  logging::LogContext ctx("LowerToPopart::loadExecutableFromFile");
  // Init the session, this also involves compiling to poplar.
  _compiler.initSession(_optimizers, getModelProtoFilename().c_str());
  _compiler.loadExecutableAndPrepareDevice(input_filename.c_str());

  std::vector<at::ScalarType> data_types;
  for (auto id : _output_tensor_hooks) {
    data_types.emplace_back(fromPopartType(_compiler.getPopartType(id)));
  }

  return std::make_shared<PoplarExecutable>(
      std::move(_compiler), std::move(_input_tensor_hooks),
      std::move(_output_tensor_hooks), std::move(data_types), _parameter_names,
      std::move(_inplace_info));
}

void LowerToPopartImpl::lower(std::vector<at::Tensor> *in_tensors) {
  logging::debug("Graph lowered to Popart {");
  // Lower the tensor parameters of the _graph to OpInputs.
  lowerParameters(in_tensors);

  // Lower the body of the _graph.
  lowerBody();

  lowerReturn();

  logging::debug("}");
  _lowered = true;
}

void LowerToPopartImpl::printWasLoweredDebug(
    const torch::jit::Node *node,
    popart_compiler::TensorId first_output_tensor) {
  logging::debug(
      "{} was lowered to {} [{},{}]", nodeToString(node),
      tensorNames(first_output_tensor, node->outputs().size()),
      tensorTypesAndShapes(first_output_tensor, node->outputs().size()),
      _compiler.getExecutionInfo().get());
}

void LowerToPopartImpl::lowerReturn() {
  // Used to encode the number of (actual) outputs
  _compiler.addOutputType(
      {popart_compiler::OutputElemType::Tuple,
       static_cast<std::int64_t>(_inplace_info.num_normal_outputs)});

  // Recursively go through the output's type to flatten its structure and
  // add it to the compiler.
  // In this representation, (T0, T1, (T2, T3), T4) would be
  // [ Tuple3, Tensor, Tensor, Tuple2, Tensor, Tensor, Tensor]

  // Only lower the outputs not used for tensors modified inplace.
  std::function<void(c10::TypePtr)> process_type;
  process_type = [this, &process_type](const c10::TypePtr &type) {
    switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      _compiler.addOutputType({popart_compiler::OutputElemType::Tensor});
      break;
    }
    case c10::TypeKind::TupleType: {
      auto tuple_type = type->expect<c10::TupleType>();
      _compiler.addOutputType(
          {popart_compiler::OutputElemType::Tuple,
           static_cast<std::int64_t>(tuple_type->elements().size())});
      for (const auto &elt_type : tuple_type->elements()) {
        process_type(elt_type);
      }
      break;
    }
    case c10::TypeKind::ListType: {
      // Use our custom type to find the number of tensors (lists can only be
      // tensors as enforced by torch JIT)

      // type->expect is static and always succeeds
      auto list_type = std::dynamic_pointer_cast<ListTypeWithNumElements>(type);
      ERROR_ON(!list_type);

      _compiler.addOutputType(
          {popart_compiler::OutputElemType::List,
           static_cast<std::int64_t>(list_type->numElements())});

      for (size_t i = 0; i < list_type->numElements(); i++) {
        _compiler.addOutputType({popart_compiler::OutputElemType::Tensor});
      }
      break;
    }
    default:
      ERROR("Unsupported output type '" << c10::typeKindToString(type->kind()));
    }
  };
  logging::debug("  return (");
  for (torch::jit::Value *value : _graph.outputs()) {
    auto tensors = _value_map.tensors(value);
    std::ostringstream ss;
    ss << "    output: %" << value->debugName() << " : " << *value->type()
       << " ->";
    logging::debug("{} {} [{}]", ss.str(), tensorNames(tensors),
                   tensorTypesAndShapes(tensors));
    if (value->type()->kind() == c10::TypeKind::ListType) {
      c10::TypeKind elt_kind =
          value->type()->expect<c10::ListType>()->getElementType()->kind();
      ERROR_ON_MSG(elt_kind != c10::TypeKind::TensorType,
                   "Unsupported list type " << c10::typeKindToString(elt_kind));
      std::int64_t num_tensors = static_cast<std::int64_t>(tensors.size());
      _compiler.addOutputType(
          {popart_compiler::OutputElemType::List, num_tensors});
      logging::trace("List with num tensors: {}", num_tensors);
      for (std::int64_t i = 0; i < num_tensors; ++i) {
        _compiler.addOutputType({popart_compiler::OutputElemType::Tensor});
      }
    } else {
      process_type(value->type());
    }

    uint64_t output_num = 0;
    for (auto id : tensors) {
      auto overlap_symbol = getOverlapSymbol("output", output_num);
      ERROR_ON(!_graph.return_node()->hasAttribute(overlap_symbol));
      auto overlap_str = _graph.return_node()->s(overlap_symbol);

      _compiler.addOutputTensor(id, popart_compiler::PopartOutputMode::N, 1,
                                overlap_str.c_str());
      _output_tensor_hooks.push_back(id);
      output_num++;
    }
  }
  logging::debug("  )");

  for (const auto &anchor : _anchors) {
    const char *name = anchor.name.c_str();
    popart_compiler::PopartOutputMode output_mode =
        static_cast<popart_compiler::PopartOutputMode>(anchor.mode);
    size_t return_period = anchor.period;

    logging::debug("  anchor ( {} {}/{} )", name,
                   outputModeToString(output_mode), return_period);

    auto id = _compiler.createTensorId(name);
    _compiler.addOutputType({popart_compiler::OutputElemType::Tensor});
    _compiler.addOutputTensor(id);
    _output_tensor_hooks.push_back(id);
  }
}

std::string LowerToPopartImpl::tensorNames(std::int64_t first_tensor,
                                           std::int64_t num_tensors) {
  ValueMap::TensorList tensors;
  tensors.reserve(num_tensors);
  for (int i = 0; i < num_tensors; i++) {
    tensors.push_back(first_tensor + i);
  }
  return tensorNames(tensors);
}
std::string
LowerToPopartImpl::tensorNames(const ValueMap::TensorList &tensors) {
  std::string sep{};
  std::string names;
  for (auto tensor : tensors) {
    names += sep + _compiler.tensorName(tensor);
    sep = ", ";
  }
  return names;
}

std::string LowerToPopartImpl::tensorTypesAndShapes(std::int64_t first_tensor,
                                                    std::int64_t num_tensors) {
  ValueMap::TensorList tensors;
  tensors.reserve(num_tensors);
  for (int i = 0; i < num_tensors; i++) {
    tensors.push_back(first_tensor + i);
  }
  return tensorTypesAndShapes(tensors);
}

std::string
LowerToPopartImpl::tensorTypesAndShapes(const ValueMap::TensorList &tensors) {
  std::string sep{};
  std::string shapes;

  const char *shape_inf_failed = "(shape inference failed)";

  for (auto tensor : tensors) {
    std::ostringstream shape_str;

    try {
      auto tensor_shape = _compiler.getSize(tensor);

      auto dtype_chars = _compiler.getTensorDTypeString(tensor);
      shape_str << dtype_chars.get();

      if (tensor_shape == popart_compiler::Compiler::invalid_size) {
        shape_str << shape_inf_failed;
      } else {
        shape_str << "(";
        for (auto it = tensor_shape.begin(); it != tensor_shape.end(); it++) {
          shape_str << *it;
          if (it + 1 != tensor_shape.end()) {
            shape_str << ", ";
          }
        }
        shape_str << ")";
      }
    } catch (const logging::Error &) {
      shape_str << shape_inf_failed;
    }

    shapes += sep + shape_str.str();
    sep = ", ";
  }
  return shapes;
}

void LowerToPopartImpl::validateOutputShapeAndType(
    popart_compiler::TensorId output_tensor, torch::jit::Node *node,
    std::uint64_t node_output) {
  torch::jit::Value *output = node->output(node_output);
  JitTensorInfo jit_output(output);

  at::ScalarType popart_type =
      fromPopartType(_compiler.getPopartType(output_tensor));
  auto popart_size = _compiler.getSize(output_tensor);
  bool match = (popart_type == jit_output.scalar_type);
  // Only validate shape if PopART's shape inference worked.
  if (match && popart_size != popart_compiler::Compiler::invalid_size) {
    match = (popart_size == jit_output.dims);
  }
  ERROR_ON_MSG(!match, "Output[" << node_output << "] mismatch: "
                                 << nodeToString(node) << " -> PopART "
                                 << tensorTypesAndShapes(output_tensor, 1));
}
// Lower the main body of the _graph.
void LowerToPopartImpl::lowerBody() {
  logging::LogContext ctx_func("LowerToPopartImpl::lowerBody");
  for (torch::jit::Node *node : _graph.nodes()) {
    logging::LogContext ctx("processing " + nodeToString(node));
    // Switch/lookup based on the actual int value.
    const c10::Symbol kind = node->kind();
    // When using the dispatcher metadata should always be set.
    std::string meta;
    if (node->sourceRange().source()) {
      meta = node->sourceRange().source()->text();
    }
    ERROR_ON_MSG(_built_in_params && meta.empty(),
                 "Source code location missing for node " + nodeToString(node));
    // Note: filename and line number might still not be available (For example
    // if the filter set by the user excludes the entire stack).
    auto file_line_col = node->sourceRange().file_line_col();
    std::uint64_t line = 0;
    std::uint64_t col = 0;
    std::string filename;
    if (file_line_col) {
      std::tie(filename, line, col) = *file_line_col;
    }
    _compiler.setCurrentPythonCodeLocation(meta.c_str(), filename.c_str(), line,
                                           col);

    auto itr = _functionToImplementation.find(kind);
    if (itr != _functionToImplementation.end()) {
      // Get the torch jit SSA for the input/output values.
      std::vector<popart_compiler::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       // Tuples aren't supported here but it's ok because
                       // we don't support any operations which actually take in
                       // tuples.
                       return _value_map.tensor(val);
                     });

      // Call the callback
      popart_compiler::TensorId first_output_tensor = itr->second(inputs, node);

      // The callback only returns the ID of the first tensor, but we know
      // the generated tensors have contiguous IDs, so we can infer the other
      // IDs.
      for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
        torch::jit::Value *output = node->output(i);
        popart_compiler::TensorId output_tensor = first_output_tensor + i;
        ERROR_ON_MSG(!_compiler.tensorIdIsValid(output_tensor),
                     "Output " << i << " doesn't exist of Node " << *node);
        // TODO(T66614): JIT graph doesn't have any shape inference so we can't
        // validate the shapes. Revisit once we've migrated to MLIR.
        // validateOutputShapeAndType(output_tensor, node, i);
        _value_map.setTensor(output, output_tensor);
      }

      if (!_compiler.isHostSideConstant(first_output_tensor)) {
        printWasLoweredDebug(node, first_output_tensor);
      }
    } else if (kind == symbols::poptorch::end_ipu_block) {
      _compiler.clearActiveIpu();
    } else if (kind == symbols::poptorch::start_for_loop) {
      _compiler.startSubgraph();
      logging::debug("{} was lowered", nodeToString(node));
    } else if (kind == symbols::poptorch::end_for_loop) {
      std::vector<popart_compiler::TensorId> inputs =
          _value_map.tensors(node->input(0));

      // Popart needs to know the number of outputs even though it's in the
      // graph.
      const std::size_t num_outputs = node->i(c10::Symbol::attr("num_outputs"));

      const std::int32_t trip_count =
          static_cast<std::int32_t>(node->i(c10::Symbol::attr("trip_count")));

      // Call the callback. This will pop the subgraphs from the stack.
      popart_compiler::TensorId first_output_tensor =
          _compiler.endForLoop(trip_count, num_outputs, inputs);

      // The callback only returns the ID of the first tensor, but we know
      // the generated tensors have contiguous IDs, so we can infer the other
      // IDs.
      std::vector<popart_compiler::TensorId> outs;
      outs.resize(num_outputs);
      for (std::uint64_t i = 0; i < num_outputs; ++i) {
        outs[i] = first_output_tensor + i;
      }

      _value_map.setTuple(node->output(), outs);
      printWasLoweredDebug(node, first_output_tensor);
    } else if (kind == symbols::poptorch::add_untyped_input_tensor) {
      popart_compiler::TensorId out = _compiler.addUntypedInputTensor();
      _value_map.setTensor(node->output(), out);
      printWasLoweredDebug(node, out);
    } else if (kind == symbols::poptorch::begin_ipu_block) {
      _compiler.setActiveIpu(node->i(c10::Symbol::attr("stage")),
                             node->i(c10::Symbol::attr("phase")),
                             node->i(c10::Symbol::attr("ipu")));
    } else if (kind == symbols::poptorch::push_name_scope) {
      _compiler.pushNameScope(node->s(c10::Symbol::attr("name")).c_str());
    } else if (kind == symbols::poptorch::pop_name_scope) {
      _compiler.popNameScope();
    } else if (kind == symbols::poptorch::set_matmul_serialization) {
      popart_compiler::TensorId input = _value_map.tensor(node->input());
      _compiler.setMatMulSerialization(
          input, node->s(c10::Symbol::attr("mode")).c_str(),
          node->i(c10::Symbol::attr("factor")),
          node->i(c10::Symbol::attr("keep_precision")));
      _value_map.setTensor(node->output(), input);
    } else if (kind == symbols::poptorch::optimizer_group) {
      std::vector<popart_compiler::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       return _value_map.tensor(val);
                     });

      std::uint64_t group = node->i(c10::Symbol::attr("group"));
      _compiler.optimizerGroup(inputs, group);

    } else if (kind == symbols::poptorch::set_available_memory) {
      // Get the torch jit SSA for the input/output values.
      std::vector<std::set<popart_compiler::TensorId>> inputs;
      for (auto *input : node->inputs()) {
        inputs.emplace_back();
        auto outputs = input->node()->outputs();
        std::transform(
            std::begin(outputs), std::end(outputs),
            std::inserter(inputs.back(), std::begin(inputs.back())),
            [&](torch::jit::Value *val) { return _value_map.tensor(val); });
      }

      _compiler.setAvailableMemoryProportion(
          inputs, node->f(c10::Symbol::attr("availableMemoryProportion")));

      for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
        _value_map.setTensor(node->output(i),
                             _value_map.tensor(node->input(i)));
      }

    } else if (kind == c10::prim::Constant) {
      ERROR_ON_MSG(node->hasAttribute(c10::attr::value),
                   "Only None constants should be left in the graph after the "
                   "CanonicaliseConstants pass");
      _value_map.setTensor(node->output(), popart_compiler::NoneTensor);
    } else if (kind == c10::prim::TupleConstruct ||
               kind == c10::prim::ListConstruct) {
      // Get the torch jit SSA for the input/output values.
      torch::jit::Value *output = node->output();

      // Add the values to the value map.
      ValueMap::TensorList input_tensors;
      for (torch::jit::Value *ids : node->inputs()) {
        for (auto tensor : _value_map.tensors(ids)) {
          input_tensors.push_back(tensor);
        }
      }
      if (kind == c10::prim::TupleConstruct) {
        _value_map.setTuple(output, input_tensors);
      } else {
        _value_map.setList(output, input_tensors);
      }
      logging::debug("{} was lowered", nodeToString(node));
    } else if (kind == c10::prim::TupleUnpack ||
               kind == c10::prim::ListUnpack) {
      // Get the torch jit SSA for the input/output values.
      const auto &tensors(_value_map.listTuple(node->input()));
      auto tensor_it = tensors.begin();
      // As tuples may be nested, walk recursively to flatten all tensors
      std::function<void(c10::TypePtr, ValueMap::TensorList &)> flattened_tuple;
      flattened_tuple = [&](const c10::TypePtr &type,
                            ValueMap::TensorList &tensorList) {
        switch (type->kind()) {
        case c10::TypeKind::TensorType: {
          ERROR_ON_MSG(tensor_it == tensors.end(),
                       "Not enough tensors to unpack");
          tensorList.push_back(*tensor_it);
          tensor_it++;
          break;
        }
        case c10::TypeKind::TupleType: {
          auto tuple = type->expect<c10::TupleType>();
          for (const auto &elt_type : tuple->elements()) {
            flattened_tuple(elt_type, tensorList);
          }
          break;
        }
        default:
          ERROR("Unsupported type '" << c10::typeKindToString(type->kind()));
        }
      };

      for (auto *output : node->outputs()) {
        switch (output->type()->kind()) {
        case c10::TypeKind::TensorType: {
          ERROR_ON(tensor_it == tensors.end());
          _value_map.setTensor(output, *tensor_it);
          tensor_it++;
          break;
        }
        case c10::TypeKind::ListType: // (should only have TensorType)
        case c10::TypeKind::TupleType: {
          ValueMap::TensorList tensor_list;
          flattened_tuple(output->type(), tensor_list);
          _value_map.setTuple(output, tensor_list);
          break;
        }
        default:
          ERROR("Unsupported parameter type '"
                << c10::typeKindToString(output->type()->kind()));
        }
      }
      ERROR_ON_MSG(tensor_it != tensors.end(), "Didn't unpack all the tensors");
      logging::debug("{} was lowered", nodeToString(node));
    } else if (kind == symbols::poptorch::host_side_cast) {
      // Map to the input value since the type will be cast host side
      ERROR_ON_MSG(!_value_map.hasTensor(node->input()),
                   "Input to host side cast has not been registered");

      ERROR_ON_MSG(node->inputs().size() != 1,
                   "Host side cast should only have one input.");

      _value_map.setTensor(node->output(), _value_map.tensor(node->input()));

    } else if (kind == symbols::poptorch::multi_conv_part) {
      std::vector<popart_compiler::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       return _value_map.tensor(val);
                     });

      _compiler.addMultiConvPart(inputs,
                                 node->is(c10::Symbol::attr("dilations")),
                                 node->is(c10::Symbol::attr("kernel_shape")),
                                 node->is(c10::Symbol::attr("pads")),
                                 node->is(c10::Symbol::attr("strides")));

      logging::debug("{} was lowered as component of MultiConv",
                     nodeToString(node));

    } else if (kind == symbols::poptorch::end_multi_conv) {
      // Extract multiconv options that are set as attributes on the
      // end_multi_conv instruction
      auto amp = c10::Symbol::attr("available_memory_proportions");
      if (node->hasAttribute(amp)) {
        _compiler.setMultiConvAvailableMemoryProportions(node->fs(amp));
      }

      auto partials_types = c10::Symbol::attr("partials_types");
      if (node->hasAttribute(partials_types)) {
        _compiler.setMultiConvPartialsTypes(node->is(partials_types));
      }

      auto conv_ditherings = c10::Symbol::attr("enable_conv_dithering");
      if (node->hasAttribute(conv_ditherings)) {
        _compiler.setMultiConvEnableConvDithering(node->is(conv_ditherings));
      }

      auto plan_type = c10::Symbol::attr("plan_type");
      if (node->hasAttribute(plan_type)) {
        _compiler.setMultiConvPlanType(node->i(plan_type));
      }

      auto per_conv_reserved_tiles =
          c10::Symbol::attr("per_conv_reserved_tiles");
      if (node->hasAttribute(per_conv_reserved_tiles)) {
        _compiler.setMultiConvPerConvReservedTiles(
            node->i(per_conv_reserved_tiles));
      }

      auto cycle_back_off = c10::Symbol::attr("cycle_back_off");
      if (node->hasAttribute(cycle_back_off)) {
        _compiler.setMultiConvCycleBackOff(node->f(cycle_back_off));
      }

      torch::jit::ArrayRef<torch::jit::Value *> node_outputs = node->outputs();
      std::vector<popart_compiler::TensorId> outputs = _compiler.endMultiConv();
      ERROR_ON_MSG(outputs.size() != node_outputs.size(),
                   "Wrong number of outputs for MultiConv. Expected "
                       << node_outputs.size() << " outputs but only received "
                       << outputs.size() << " outputs.");

      for (size_t i = 0; i < outputs.size(); i++) {
        _value_map.setTensor(node_outputs[i], outputs[i]);
      }

      printWasLoweredDebug(node, outputs.front());
    } else if (kind == symbols::poptorch::canonicalised_cpu_call) {
      // CPU callbacks are referenced by an string identifier.
      std::string id = node->s(c10::Symbol::attr("ID"));

      std::vector<popart_compiler::PopartType> input_types;
      std::vector<std::vector<std::size_t>> input_shapes;

      // Get the torch jit SSA for the input/output values.
      std::vector<popart_compiler::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       // Append type info from the inputs.
                       platformAgnosticTypeInfoFromIRType(val, &input_types,
                                                          &input_shapes);

                       return _value_map.tensor(val);
                     });

      std::vector<popart_compiler::PopartType> output_types;
      std::vector<std::vector<std::size_t>> output_shapes;

      for (torch::jit::Value *value : node->outputs()) {
        platformAgnosticTypeInfoFromIRType(value, &output_types,
                                           &output_shapes);
      }

      popart_compiler::TensorId first_output_tensor =
          _compiler.addCPUCallback(inputs, _callbacks[id], input_types,
                                   input_shapes, output_types, output_shapes);

      for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
        torch::jit::Value *output = node->output(i);
        popart_compiler::TensorId output_tensor = first_output_tensor + i;
        ERROR_ON_MSG(!_compiler.tensorIdIsValid(output_tensor),
                     "Output " << i << " doesn't exist of Node " << *node);
        _value_map.setTensor(output, output_tensor);
      }
      printWasLoweredDebug(node, first_output_tensor);
    } else if (kind == symbols::poptorch::set_attribute) {
      const std::string &attribute = node->s(c10::Symbol::attr("attribute"));
      const std::string &key = node->s(c10::Symbol::attr("key"));
      const std::string &value = node->s(c10::Symbol::attr("value"));
      _compiler.setAttribute(attribute.c_str(), key.c_str(), value.c_str());
    } else if (kind == symbols::poptorch::clear_attribute) {
      const std::string &attribute = node->s(c10::Symbol::attr("attribute"));
      const std::string &key = node->s(c10::Symbol::attr("key"));
      _compiler.clearAttribute(attribute.c_str(), key.c_str());
    } else {
      ERROR("Couldn't find a registered operation for node " << *node);
    }
  }
}

void LowerToPopartImpl::lowerParameters(std::vector<at::Tensor> *in_tensors) {
  // The "true" inputs are a mixture of tuples (which may be nested) and tensors
  // The parameters are all tensors. "_graph.inputs()." contains the inputs
  // first followed by the parameters at the end.

  // This will provide a view of all the tensors in _graph.inputs(), i.e.
  // by collapsing tuples.
  auto graph_t_inputs = collapsedGraphInputHierachy(&_graph);

  std::function<bool(torch::jit::Value * value)> is_parameter;
  std::function<bool(torch::jit::Value * value)> is_parameter_tensor;
  std::function<void *(torch::jit::Value * value)> get_data_ptr;
  std::function<std::string(torch::jit::Value * value)> get_parameter_name;
  std::function<JitTensorInfo(torch::jit::Value * value)> get_type_info;

  if (_built_in_params) {
    ERROR_ON_MSG(in_tensors != nullptr,
                 "[Internal] Inputs expected to be in the graph.");
    // Bind to the DispatchTracer functions.
    is_parameter = isParameter;
    is_parameter_tensor = isParameter;
    get_data_ptr = getDataSourceForValue;
    get_parameter_name = [](torch::jit::Value *value) {
      auto name = getParameterName(value);
      ERROR_ON_MSG(name.empty(), "No parameter name available for value %"
                                     << value->debugName());
      return name;
    };
    get_type_info = [](torch::jit::Value *value) {
      return JitTensorInfo(value);
    };

    // Step 0, remove unused parameters
    // graph_t_inputs is updated but _graph.inputs() will retain unused
    // parameters
    std::vector<bool> parameter_used(graph_t_inputs.size(), true);
    for (size_t i = 0; i < graph_t_inputs.size(); ++i) {
      auto *value = graph_t_inputs[i];
      if (value->uses().empty() && is_parameter(value)) {
        parameter_used.at(i) = false;
        logging::trace("Skipping unused parameter: %{}", value->debugName());
      }
    }
    maskVector(&graph_t_inputs, parameter_used);
  } else {
    // We're using tracing: the parameters have been provided separately
    // and the graph still needs to be cleaned up.
    std::size_t num_params = _parameters.size();
    const size_t num_t_inputs = graph_t_inputs.size() - num_params;
    // in_tensors contains both "true" inputs and parameters.
    ERROR_ON(in_tensors == nullptr);
    ERROR_ON(in_tensors->size() != num_t_inputs);
    ERROR_ON(graph_t_inputs.size() != (in_tensors->size() + num_params));

    // Step 0, remove unused parameters
    // graph_t_inputs is updated but _graph.inputs() will retain unused
    // parameters
    std::vector<bool> parameter_used(_parameters.size(), true);
    for (size_t index = 0; index < _parameters.size(); index++) {
      ERROR_ON(!parameter_used.at(index));
      auto *value = graph_t_inputs[num_t_inputs + index];
      if (value->uses().empty()) {
        parameter_used.at(index) = false;

        logging::trace("Skipping unused parameter: {}",
                       _parameter_names.at(index));
      }
    }

    size_t num_inputs = _graph.inputs().size() - _parameters.size();

    maskVector(&graph_t_inputs, parameter_used, num_t_inputs);
    // Use remove-erase idiom to remove parameters with linear complexity
    maskVector(&_parameters, parameter_used);
    maskVector(&_parameter_names, parameter_used);
    ERROR_ON(num_t_inputs + _parameters.size() != graph_t_inputs.size());

    std::set<torch::jit::Value *> input_values;
    for (size_t i = 0; i < num_inputs; ++i) {
      input_values.emplace(_graph.inputs().at(i));
    }

    // value from _graph.inputs()
    is_parameter = [=](torch::jit::Value *value) {
      return input_values.count(value) == 0;
    };

    std::map<torch::jit::Value *, size_t> index_map_tensors;
    for (size_t i = 0; i < graph_t_inputs.size(); ++i) {
      index_map_tensors.emplace(graph_t_inputs[i], i);
    }

    // value from graph_t_inputs
    auto index_of_tensor = [=](torch::jit::Value *value) {
      return index_map_tensors.at(value);
    };

    is_parameter_tensor = [=](torch::jit::Value *value) {
      return index_of_tensor(value) >= num_t_inputs;
    };

    // value comes from graph_t_inputs
    get_data_ptr = [=](torch::jit::Value *value) {
      auto index = index_of_tensor(value);
      at::Tensor &tensor(is_parameter_tensor(value)
                             ? _parameters.at(index - num_t_inputs)
                             : (*in_tensors)[index]);
      return tensor.data_ptr();
    };

    get_type_info = [=](torch::jit::Value *value) {
      auto index = index_of_tensor(value);
      at::Tensor &tensor(is_parameter_tensor(value)
                             ? _parameters.at(index - num_t_inputs)
                             : (*in_tensors)[index]);
      return JitTensorInfo(tensor);
    };
  }

  // Step 1, add tensor inputs for all tensors in the hierarchy and obtain
  // the resulting popart IDs. This can be done with collapsed hierarchy.
  ValueMap::TensorList parameter_popart_ids;
  std::vector<torch::jit::Value *> parameter_values;
  size_t input_index = 0;
  size_t param_index = 0;
  for (auto *value : graph_t_inputs) {
    JitTensorInfo info = get_type_info(value);
    std::string popart_type = typeToPopartStr(info.scalar_type);
    if (is_parameter_tensor(value)) {
      void *data_ptr = get_data_ptr(value);
      ERROR_ON_MSG(value->uses().empty(),
                   "Parameter %"
                       << value->debugName()
                       << " isn't used and therefore should have been removed");
      ERROR_ON(param_index > _parameter_names.size());
      // If the parameter names were not populated ahead of time
      // generate / extract them from the value now.
      if (param_index == _parameter_names.size()) {
        ERROR_ON_MSG(!get_parameter_name,
                     "Name for parameter "
                         << param_index << " requested but only have "
                         << _parameter_names.size() << " names.");
        _parameter_names.push_back(get_parameter_name(value));
      }

      std::string name = _parameter_names.at(param_index);
      auto id = _compiler.addInitializedInputTensor(
          name.c_str(), popart_type.c_str(), info.dims, data_ptr);
      parameter_values.push_back(value);
      parameter_popart_ids.push_back(id);
      param_index++;
    } else {
      auto overlap_symbol = getOverlapSymbol("input", input_index);
      std::string overlap_str("no_overlap");
      if (_graph.param_node()->hasAttribute(overlap_symbol)) {
        overlap_str = _graph.param_node()->s(overlap_symbol);
      }

      auto id = _compiler.addInputTensor(popart_type.c_str(), info.dims,
                                         overlap_str.c_str());
      _input_tensor_hooks.push_back(id);
      input_index++;
    }
  }

  if (!_built_in_params) {
    ERROR_ON(parameter_popart_ids.size() != _parameters.size());
  }

  // Step 2, map the PopART tensor IDs to the JIT Value of the (not collapsed)
  // graph inputs
  logging::debug("graph(");
  auto input_tensor_it = _input_tensor_hooks.begin();
  size_t index = 0;
  for (torch::jit::Value *value : _graph.inputs()) {
    if (is_parameter(value)) {
      // Only process inputs
      continue;
    }
    ERROR_ON(value->node()->kind() != c10::prim::Param);
    size_t num_tensors = numTensorsForType(value->type());

    ValueMap::TensorList tensors;
    tensors.reserve(num_tensors);

    for (size_t i = 0; i < num_tensors; i++) {
      ERROR_ON(input_tensor_it == _input_tensor_hooks.end());
      tensors.push_back(*input_tensor_it);
      input_tensor_it++;
    }

    if (value->type()->kind() == c10::TypeKind::TensorType) {
      ERROR_ON(tensors.size() != 1);
      _value_map.setTensor(value, tensors.front());
    } else {
      ERROR_ON(value->type()->kind() != c10::TypeKind::TupleType);
      _value_map.setTuple(value, tensors);
    }

    std::ostringstream ss;
    ss << "      input: %" << value->debugName() << " : " << *value->type()
       << " ->";
    logging::debug("{} {} [{}]", ss.str(), tensorNames(tensors),
                   tensorTypesAndShapes(tensors));

    index++;
  }

  // Step 3, map the PopART tensor IDs to the JIT Value of the parameters
  for (index = 0; index < parameter_popart_ids.size(); index++) {
    auto *value = parameter_values.at(index);
    auto &tensor(parameter_popart_ids.at(index));

    std::ostringstream ss;
    ss << "      param: %" << value->debugName() << " : " << *value->type()
       << " ->";
    logging::debug("{} {} [{}]", ss.str(), tensorNames(tensor, 1),
                   tensorTypesAndShapes(tensor, 1));
    _value_map.setTensor(value, tensor);
  }
  logging::debug("  ):");
}

namespace {
// Helper to let us filter string arguments into const char*s. This is to catch
// the std::string produced by some attributes before they cross the ABI
// boundary.

template <typename T> T convertType(T &&t) { return t; }

// String, return const char*.
const char *convertType(const std::string &s) {
  return s.c_str(); // NOLINT
}

// vector<string>, return vector<const char*>
std::vector<const char *> convertType(const std::vector<std::string> &s) {
  std::vector<const char *> result;
  std::transform(s.begin(), s.end(), std::back_inserter(result),
                 [](const std::string &str) {
                   return str.c_str(); // NOLINT
                 });
  return result;
}

// vector<double, return vector<float>
std::vector<float> convertType(const std::vector<double> &v) {
  std::vector<float> result;
  std::transform(v.begin(), v.end(), std::back_inserter(result),
                 [](double d) { return static_cast<float>(d); });
  return result;
}

popart_compiler::PopartConstant
convertTensorConstantNode(const torch::jit::Node *node) {
  logging::LogContext ctx("convertTensorConstantNode: processing " +
                          nodeToString(node));

  ERROR_ON_MSG(
      node->kind() != symbols::poptorch::tensor_constant,
      "Only a popart_compiler::tensor_constant can be converted into a popart "
      "constant");
  auto output_type =
      *node->output()->type()->expect<c10::TensorType>()->scalarType();
  auto tensor_type = getNodeTensorAttrValue(node).scalar_type();

  ERROR_ON_MSG(output_type != tensor_type, "Output type is "
                                               << c10::toString(output_type)
                                               << " but tensor type is "
                                               << c10::toString(tensor_type));

  auto tensor = getNodeTensorAttrValue(node);
  ERROR_ON(!tensor.is_contiguous());

  return {toPopartType(tensor.scalar_type()), tensor.data_ptr(),
          getTensorDimensions(tensor)};
}

popart_compiler::HostSideConstant
convertHostSideTensorConstantNode(const torch::jit::Node *node) {
  logging::LogContext ctx("convertHostSideTensorConstantNode: processing " +
                          nodeToString(node));
  ERROR_ON_MSG(node->kind() != symbols::poptorch::host_side_tensor_constant,
               "Only a poptorch::host_side_tensor_constant can be converted "
               "into a host side constant constant");

  auto tensor = getNodeTensorAttrValue(node);
  ERROR_ON(!tensor.is_contiguous());

  return {toPopartType(tensor.scalar_type()), tensor.data_ptr(),
          tensor.nbytes(), getTensorDimensions(tensor)};
}

void processListAttribute(
    const char *name,
    const std::shared_ptr<std::vector<popart_compiler::PopartAttribute>>
        &attributes,
    const IPyValue &elements) {
  const auto first_element = elements.getFromList(0);

  if (first_element->isInt()) {
    std::vector<int64_t> ints;
    ints.reserve(elements.getListSize());
    elements.forEachInList([&ints](const IPyValue &int_obj) {
      ints.push_back(int_obj.toInt64());
    });
    attributes->emplace_back(name, ints);
    return;
  }

  if (first_element->isDouble()) {
    std::vector<float> floats;
    floats.reserve(elements.getListSize());
    elements.forEachInList([&floats](const IPyValue &float_obj) {
      floats.push_back(float_obj.toFloatWithRangeCheck());
    });
    attributes->emplace_back(name, floats);
    return;
  }

  if (first_element->isString()) {
    std::vector<std::unique_ptr<char[]>> strs;
    strs.reserve(elements.getListSize());
    elements.forEachInList([&strs](const IPyValue &str) {
      strs.emplace_back(stringToUniquePtr(str.toString()));
    });
    attributes->emplace_back(name, strs);
    return;
  }

  ERROR("Invalid type for Popart attribute.");
}

std::shared_ptr<std::vector<popart_compiler::PopartAttribute>>
convertCustomOpAttributes(const torch::jit::Node *node,
                          const AttributeAccessor &attribute_accessor) {
  logging::LogContext ctx("convertCustomOpAttributes: processing " +
                          nodeToString(node));
  std::string attributes_id_str(node->s(c10::Symbol::attr("attributes_id")));

  auto dict_obj = attribute_accessor(attributes_id_str);
  auto attributes =
      std::make_shared<std::vector<popart_compiler::PopartAttribute>>();
  dict_obj->forEachInDict([&attributes](const IPyValue &key,
                                        const IPyValue &attribute) {
    auto name = key.toString();

    if (attribute.isInt()) {
      attributes->emplace_back(name.c_str(), attribute.toInt64());
    } else if (attribute.isDouble()) {
      attributes->emplace_back(name.c_str(), attribute.toFloatWithRangeCheck());
    } else if (attribute.isString()) {
      attributes->emplace_back(name.c_str(),
                               stringToUniquePtr(attribute.toString()));
    } else if (attribute.isSetListOrTuple()) {
      processListAttribute(name.c_str(), attributes, attribute);
    } else {
      ERROR("Invalid attribute type");
    }
  });

  return attributes;
}
} // namespace

LowerToPopartImpl::LowerToPopartImpl(
    torch::jit::Graph *g, InplaceGraphInfo &&inplace_info, bool training,
    std::vector<popart_compiler::Optimizer> &&opt,
    const popart_compiler::SessionOptions &options,
    const AttributeAccessor &attribute_accessor, CPUCallbackMap &&callback,
    const AnchorList &&anchors)
    : _graph(*g), _lowered(false), _built_in_params(true),
      _inplace_info(std::move(inplace_info)), _optimizers(opt),
      _anchors(anchors), _compiler(training, options), _callbacks(callback) {
  // Init the function implementation map. This map will be populated by
  // elements which look something like:
  /* {"popart::Foo", [&](const std::vector<popart_compiler::TensorId> &inputs,
     torch::jit::Node *node) { return _compiler.foo(inputs,
          node->i("attr::SomeIntegerAttr"),
    node->i("attr::SomeOtherIntegerAttr"), node->is("attr::AnIntArrayAttr"),
    node->f("attr::AFloatAttr"));
      }
    },
  */
  // Essentially this is just a map from the string IR symbol to a function to
  // be called that implements it. Those functions are also autogenerated by the
  // same macros in _compiler.hpp and _compiler.cpp.
  _functionToImplementation = {
// Torch JIT api defines the attribute accessor as the following function names.
#define INT_VEC is
#define FLOAT_VEC fs
#define FLOAT f
#define INT i
#define CHAR i
#define BOOL i
#define STRING s
#define STRING_VEC ss

// Useful NOP macro
#define NONE

// The arguments are processed by extracting the given type using the above
// accessors, the name is converted into "attr::NAME" which is what pytorch JIT
// expects for attribute accessing.
#define ARG(Type, Name) , convertType(node->Type(c10::Symbol::attr(#Name)))

#define POPART_CONST_ARG(unused) , convertTensorConstantNode(node)
#define HOST_SIDE_CONST_ARG(unused)                                            \
  , std::move(convertHostSideTensorConstantNode(node))

#define POPART_ATTRIB_VEC_ARG(unused)                                          \
  , convertCustomOpAttributes(node, attribute_accessor)

#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(ns, symbolName, function, unused, Args, unused2)               \
  {symbols::ns::symbolName,                                                    \
   [&](const std::vector<popart_compiler::TensorId> &inputs,                   \
       torch::jit::Node *node) {                                               \
     (void)(node);                                                             \
     return _compiler.function(inputs Args);                                   \
   }},

#define OP_DECL_NO_RETURN(ns, symbolName, function, unused, Args, unused2)     \
  {symbols::ns::symbolName,                                                    \
   [&](const std::vector<popart_compiler::TensorId> &inputs,                   \
       torch::jit::Node *node) {                                               \
     _compiler.function(inputs Args);                                          \
     ERROR_ON_MSG(node->outputs().size() != 0,                                 \
                  "Void return function called on torch::jit::Node which has " \
                  "outputs");                                                  \
     return popart_compiler::TensorId{};                                       \
   }},

#include "popart_compiler/SupportedOperations.inc.hpp"

#undef BODY_STR_ARG
#undef STR_ARG
#undef BODY_ARG
#undef POPART_ATTRIB_VEC_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef OP_DECL
#undef OP_DECL_NO_RETURN
#undef ARG
#undef NONE
#undef BOOL
#undef CHAR
#undef STRING
#undef STRING_VEC
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC
  }; // End map initalizer.
}
} // namespace detail

LowerToPopart::LowerToPopart(torch::jit::Graph *graph,
                             const std::vector<at::Tensor> &parameters,
                             const std::vector<std::string> &parameter_names,
                             InplaceGraphInfo &&inplace_info, bool training,
                             std::vector<popart_compiler::Optimizer> &&opt,
                             const popart_compiler::SessionOptions &options,
                             const AttributeAccessor &attribute_accessor,
                             CPUCallbackMap callbacks, AnchorList &&anchors) {
  _impl = std::make_unique<detail::LowerToPopartImpl>(
      graph, std::move(inplace_info), training, std::move(opt),
      std::move(options), attribute_accessor, std::move(callbacks),
      std::move(anchors));
  _impl->setParameters(parameters, parameter_names);
}

LowerToPopart::LowerToPopart(torch::jit::Graph *graph,
                             InplaceGraphInfo &&inplace_info, bool training,
                             std::vector<popart_compiler::Optimizer> &&opt,
                             const popart_compiler::SessionOptions &options,
                             const AttributeAccessor &attribute_accessor,
                             CPUCallbackMap callbacks, AnchorList &&anchors) {
  _impl = std::make_unique<detail::LowerToPopartImpl>(
      graph, std::move(inplace_info), training, std::move(opt),
      std::move(options), attribute_accessor, std::move(callbacks),
      std::move(anchors));
}
void LowerToPopart::lower(std::vector<at::Tensor> *in_tensors) {
  _impl->lower(in_tensors);
}

std::shared_ptr<PoplarExecutable> LowerToPopart::compile() {
  auto executable = _impl->compile();
  if (logging::outputPopartIR()) {
    logging::debug("Popart IR: {}", executable->getPopartIR());
  }
  return executable;
}

std::shared_ptr<PoplarExecutable>
LowerToPopart::loadExecutableFromFile(const std::string &input_filename) {
  return _impl->loadExecutableFromFile(input_filename);
}

LowerToPopart::~LowerToPopart() = default;

LowerToPopart::LowerToPopart(LowerToPopart &&lower) {
  _impl = std::move(lower._impl);
}

} // namespace poptorch
