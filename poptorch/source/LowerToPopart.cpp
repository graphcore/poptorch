// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/LowerToPopart.hpp"

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

#include "poptorch/InplaceOps.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

// Mapping between the SSA values of torch jit with the ssa values of popart.
// Each Value is either a single tensor, tuple or list (Note: nested tuples are
// stored flattened).
class ValueMap {
public:
  using TensorList = std::vector<poptorch::TensorId>;

  poptorch::TensorId tensor(torch::jit::Value *value) const;
  const TensorList &listTuple(torch::jit::Value *value) const;

  // Return the list of tensors without checking if it's a tuple, list or a
  // single tensor.
  const TensorList &tensors(torch::jit::Value *value) const;

  bool hasTensor(torch::jit::Value *value) const {
    return _map.count(value) == 1;
  }

  void setTensor(torch::jit::Value *value, poptorch::TensorId id);
  void setList(torch::jit::Value *value, const TensorList &tensors);
  void setTuple(torch::jit::Value *value, const TensorList &tensors);

private:
  struct Data {
    explicit Data(poptorch::TensorId id) : type(OutputElemType::Tensor) {
      tensors.push_back(id);
    }

    Data(TensorList tuple, OutputElemType type_)
        : type(type_), tensors(std::move(tuple)) {}
    OutputElemType type;
    TensorList tensors;
  };
  std::unordered_map<torch::jit::Value *, Data> _map;
};

poptorch::TensorId ValueMap::tensor(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  ERROR_ON_MSG(it->second.type != OutputElemType::Tensor,
               value->debugName() << " is not a tensor");
  ERROR_ON(it->second.tensors.size() != 1);
  return it->second.tensors.front();
}

const ValueMap::TensorList &
ValueMap::listTuple(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  ERROR_ON_MSG((it->second.type != OutputElemType::Tuple &&
                it->second.type != OutputElemType::List),
               value->debugName() << " is not a tuple or list");
  return it->second.tensors;
}

const ValueMap::TensorList &ValueMap::tensors(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  return it->second.tensors;
}

void ValueMap::setTensor(torch::jit::Value *value, poptorch::TensorId id) {
  ERROR_ON_MSG(!_map.emplace(value, Data(id)).second,
               "Value " << value->debugName() << " already present in the map");
}

void ValueMap::setList(torch::jit::Value *value,
                       const ValueMap::TensorList &tensors) {
  ERROR_ON_MSG(!_map.emplace(value, Data(tensors, OutputElemType::List)).second,
               "Value " << value->debugName() << " already present in the map");
}

void ValueMap::setTuple(torch::jit::Value *value,
                        const ValueMap::TensorList &tensors) {
  ERROR_ON_MSG(
      !_map.emplace(value, Data(tensors, OutputElemType::Tuple)).second,
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

at::ScalarType fromPopartType(const poptorch::PopartType type) {
  switch (type) {
  case poptorch::PopartType::UINT8: {
    return at::ScalarType::Byte;
  }
  case poptorch::PopartType::INT8: {
    return at::ScalarType::Char;
  }
  case poptorch::PopartType::INT16: {
    return at::ScalarType::Short;
  }
  case poptorch::PopartType::INT32:
  case poptorch::PopartType::UINT32: {
    return at::ScalarType::Int;
  }
  case poptorch::PopartType::INT64: {
    return at::ScalarType::Long;
  }
  case poptorch::PopartType::BOOL: {
    return at::ScalarType::Bool;
  }
  case poptorch::PopartType::FLOAT: {
    return at::ScalarType::Float;
  }
  case poptorch::PopartType::FLOAT16: {
    return at::ScalarType::Half;
  }
  case poptorch::PopartType::BFLOAT16: {
    return at::ScalarType::BFloat16;
  }
  case poptorch::PopartType::DOUBLE: {
    return at::ScalarType::Double;
  }
  case poptorch::PopartType::COMPLEX64: {
    return at::ScalarType::ComplexFloat;
  }
  case poptorch::PopartType::COMPLEX128: {
    return at::ScalarType::ComplexDouble;
  }
  default:
    ERROR("Unsupported PopART data type");
  }
}

PopartType toPopartType(const at::ScalarType type) {
  switch (type) {
  case at::ScalarType::Byte: {
    return PopartType::UINT8;
  }
  case at::ScalarType::Char: {
    return PopartType::INT8;
  }
  case at::ScalarType::Short: {
    return PopartType::INT16;
  }
  case at::ScalarType::Int: {
    return PopartType::INT32;
  }
  case at::ScalarType::Long: {
    return PopartType::INT64;
  }
  case at::ScalarType::Bool: {
    return PopartType::BOOL;
  }
  case at::ScalarType::Float: {
    return PopartType::FLOAT;
  }
  case at::ScalarType::Half: {
    return PopartType::FLOAT16;
  }
  case at::ScalarType::BFloat16: {
    return PopartType::BFLOAT16;
  }
  case at::ScalarType::Double: {
    return PopartType::DOUBLE;
  }
  case at::ScalarType::ComplexFloat: {
    return PopartType::COMPLEX64;
  }
  case at::ScalarType::ComplexDouble: {
    return PopartType::COMPLEX128;
  }
  default:
    ERROR("Unsupported PyTorch scalar type");
  }
}

void platformAgnosticTypeInfoFromIRType(
    torch::jit::Value *value, std::vector<poptorch::PopartType> *types,
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
  LowerToPopartImpl(torch::jit::Graph *g, std::vector<at::Tensor> params,
                    std::vector<std::string> parameter_names,
                    std::shared_ptr<InplaceOpHandler> inplace_op_handler,
                    bool training, std::vector<Optimizer> &&opt,
                    const SessionOptions &options,
                    const py::function &attribute_accessor,
                    CPUCallbackMap &&callback, const AnchorList &&anchors);

  void lower(std::vector<at::Tensor> *in_tensors);

  std::shared_ptr<poptorch::PoplarExecutable> compile();
  void compileAndExport(const std::string &export_filename);
  std::shared_ptr<poptorch::PoplarExecutable>
  loadExecutableFromFile(const std::string &input_filename,
                         std::int64_t offset);

private:
  torch::jit::Graph &_graph;

  bool _lowered;

  std::vector<at::Tensor> _parameters;
  std::vector<std::string> _parameter_names;
  std::shared_ptr<InplaceOpHandler> _inplace_op_handler;

  std::vector<poptorch::TensorId> _input_tensor_hooks;

  std::vector<poptorch::TensorId> _output_tensor_hooks;

  ValueMap _value_map;

  // Optimizer from the user.
  const std::vector<Optimizer> _optimizers;

  // Tensors to be anchored other than outputs
  const AnchorList &_anchors;

  using FunctionType = std::function<poptorch::TensorId(
      const std::vector<poptorch::TensorId> &inputs, torch::jit::Node *)>;
  std::unordered_map<c10::Symbol, FunctionType> _functionToImplementation;

  poptorch::Compiler _compiler;

  CPUCallbackMap _callbacks;

  void lowerParameters(std::vector<at::Tensor> *in_tensors);

  void lowerBody();

  void lowerReturn();

  std::string tensorNames(std::int64_t first_tensor, std::int64_t num_tensors);
  std::string tensorNames(const ValueMap::TensorList &tensors);

  std::string tensorTypesAndShapes(std::int64_t first_tensor,
                                   std::int64_t num_tensors);
  std::string tensorTypesAndShapes(const ValueMap::TensorList &tensors);
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

/*
 * Lower to popart impl.
 */
std::shared_ptr<poptorch::PoplarExecutable> LowerToPopartImpl::compile() {
  ERROR_ON_MSG(!_lowered, "You need to lower() the graph first");

  logging::LogContext ctx("LowerToPopart::compile ");
  // Init the session, this also involves compiling to poplar.
  _compiler.initSession(_optimizers);

  _compiler.compileAndPrepareDevice();

  std::vector<at::ScalarType> data_types;
  for (auto id : _output_tensor_hooks) {
    data_types.emplace_back(fromPopartType(_compiler.getPopartType(id)));
  }

  return std::make_shared<poptorch::PoplarExecutable>(
      std::move(_compiler), std::move(_input_tensor_hooks),
      std::move(_output_tensor_hooks), std::move(data_types), _parameter_names,
      _inplace_op_handler);
}

std::shared_ptr<poptorch::PoplarExecutable>
LowerToPopartImpl::loadExecutableFromFile(const std::string &input_filename,
                                          std::int64_t offset) {
  logging::LogContext ctx("LowerToPopart::loadExecutableFromFile ");
  // Init the session, this also involves compiling to poplar.
  _compiler.initSession(_optimizers);
  _compiler.loadExecutableAndPrepareDevice(input_filename.c_str(), offset);

  std::vector<at::ScalarType> data_types;
  for (auto id : _output_tensor_hooks) {
    data_types.emplace_back(fromPopartType(_compiler.getPopartType(id)));
  }

  return std::make_shared<poptorch::PoplarExecutable>(
      std::move(_compiler), std::move(_input_tensor_hooks),
      std::move(_output_tensor_hooks), std::move(data_types), _parameter_names,
      _inplace_op_handler);
}

void LowerToPopartImpl::compileAndExport(const std::string &export_filename) {
  ERROR_ON_MSG(!_lowered, "You need to lower() the graph first");

  logging::LogContext ctx("LowerToPopart::compileAndExport ");
  _compiler.initSession(_optimizers);
  _compiler.compileAndExport(export_filename.c_str());
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

void LowerToPopartImpl::lowerReturn() {
  // Used to encode the number of (actual) outputs
  _compiler.addOutputType(
      {OutputElemType::Tuple,
       static_cast<std::int64_t>(_inplace_op_handler->getNumNormalOutputs())});

  // Recursively go through the output's type to flatten its structure and
  // add it to the compiler.
  // In this representation, (T0, T1, (T2, T3), T4) would be
  // [ Tuple3, Tensor, Tensor, Tuple2, Tensor, Tensor, Tensor]

  // Only lower the outputs not used for tensors modified inplace.
  size_t num_added = 0;

  std::function<void(c10::TypePtr)> process_type;
  process_type = [this, &process_type, &num_added](const c10::TypePtr &type) {
    switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      _compiler.addOutputType({OutputElemType::Tensor});
      break;
    }
    case c10::TypeKind::TupleType: {
      auto tuple_type = type->expect<c10::TupleType>();
      _compiler.addOutputType(
          {OutputElemType::Tuple,
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
          {OutputElemType::List,
           static_cast<std::int64_t>(list_type->numElements())});

      for (size_t i = 0; i < list_type->numElements(); i++) {
        _compiler.addOutputType({OutputElemType::Tensor});
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
      _compiler.addOutputType({OutputElemType::List, num_tensors});
      logging::trace("List with num tensors: {}", num_tensors);
      for (std::int64_t i = 0; i < num_tensors; ++i) {
        _compiler.addOutputType({OutputElemType::Tensor});
      }
    } else {
      process_type(value->type());
    }
    for (auto id : tensors) {
      _compiler.addOutputTensor(id);
      _output_tensor_hooks.push_back(id);
    }
  }
  logging::debug("  )");

  for (const auto &anchor : _anchors) {
    const char *name = anchor.name.c_str();
    PopartAnchorTypes anchor_mode = static_cast<PopartAnchorTypes>(anchor.mode);
    size_t return_period = anchor.period;

    logging::debug("  anchor ( {} {}/{} )", name,
                   anchorTypeToString(anchor_mode), return_period);

    auto id = _compiler.createTensorId(name);
    _compiler.addOutputType({OutputElemType::Tensor});
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

      if (tensor_shape.empty()) {
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

// Lower the main body of the _graph.
void LowerToPopartImpl::lowerBody() {
  for (torch::jit::Node *node : _graph.nodes()) {
    logging::LogContext ctx("LowerToPopartImpl::lowerBody processing " +
                            nodeToString(node));
    // Switch/lookup based on the actual int value.
    const c10::Symbol kind = node->kind();

    auto itr = _functionToImplementation.find(kind);
    if (itr != _functionToImplementation.end()) {
      // Get the torch jit SSA for the input/output values.
      std::vector<poptorch::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       // Tuples aren't supported here but it's ok because
                       // we don't support any operations which actually take in
                       // tuples.
                       return _value_map.tensor(val);
                     });

      // Call the callback
      poptorch::TensorId first_output_tensor = itr->second(inputs, node);

      // The callback only returns the ID of the first tensor, but we know
      // the generated tensors have contiguous IDs, so we can infer the other
      // IDs.
      for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
        torch::jit::Value *output = node->output(i);
        poptorch::TensorId output_tensor = first_output_tensor + i;
        ERROR_ON_MSG(!_compiler.tensorIdIsValid(output_tensor),
                     "Output " << i << " doesn't exist of Node " << *node);
        _value_map.setTensor(output, output_tensor);
      }

      if (!_compiler.isHostSideConstant(first_output_tensor)) {
        logging::debug(
            "  {} was lowered to {} [{},{}]", nodeToString(node),
            tensorNames(first_output_tensor, node->outputs().size()),
            tensorTypesAndShapes(first_output_tensor, node->outputs().size()),
            _compiler.getExecutionInfo().get());
      }
    } else if (kind == symbols::poptorch::end_ipu_block) {
      _compiler.clearActiveIpu();
    } else if (kind == symbols::poptorch::start_if_true) {
      // Starting the if block means changing the internal builder state to work
      // with a new subgraph.
      _compiler.startIfBlock();
    } else if (kind == symbols::poptorch::start_if_false) {
      // Starting the else block means changing the internal builder state to
      // work with a new subgraph.
      _compiler.startElseBlock();
    } else if (kind == symbols::poptorch::end_if) {
      // Process the if condition.
      poptorch::TensorId condition = _value_map.tensor(node->input(0));

      // Popart needs to know the number of outputs even though it's in the
      // graph.
      const std::size_t num_outputs =
          node->i(c10::Symbol::fromQualString("attr::num_outputs"));

      // Call the callback. This will pop the subgraphs from the stack.
      poptorch::TensorId first_output_tensor =
          _compiler.endIf(condition, num_outputs);

      // The callback only returns the ID of the first tensor, but we know
      // the generated tensors have contiguous IDs, so we can infer the other
      // IDs.
      std::vector<poptorch::TensorId> outs;
      outs.resize(num_outputs);
      for (std::uint64_t i = 0; i < num_outputs; ++i) {
        outs[i] = first_output_tensor + i;
      }

      _value_map.setTuple(node->output(), outs);

    } else if (kind == symbols::poptorch::start_for_loop) {
      _compiler.startSubgraph();
    } else if (kind == symbols::poptorch::end_for_loop) {
      // Process the if condition.
      std::vector<poptorch::TensorId> inputs =
          _value_map.tensors(node->input(0));

      // Popart needs to know the number of outputs even though it's in the
      // graph.
      const std::size_t num_outputs =
          node->i(c10::Symbol::fromQualString("attr::num_outputs"));

      const std::int32_t trip_count = static_cast<std::int32_t>(
          node->i(c10::Symbol::fromQualString("attr::trip_count")));

      // Call the callback. This will pop the subgraphs from the stack.
      poptorch::TensorId first_output_tensor =
          _compiler.endForLoop(trip_count, num_outputs, inputs);

      // The callback only returns the ID of the first tensor, but we know
      // the generated tensors have contiguous IDs, so we can infer the other
      // IDs.
      std::vector<poptorch::TensorId> outs;
      outs.resize(num_outputs);
      for (std::uint64_t i = 0; i < num_outputs; ++i) {
        outs[i] = first_output_tensor + i;
      }

      _value_map.setTuple(node->output(), outs);
    } else if (kind == symbols::poptorch::add_untyped_input_tensor) {
      poptorch::TensorId out = _compiler.addUntypedInputTensor();
      _value_map.setTensor(node->output(), out);
    } else if (kind == symbols::poptorch::begin_ipu_block) {
      _compiler.setActiveIpu(
          node->i(c10::Symbol::fromQualString("attr::stage")),
          node->i(c10::Symbol::fromQualString("attr::phase")),
          node->i(c10::Symbol::fromQualString("attr::ipu")));
    } else if (kind == symbols::poptorch::push_name_scope) {
      _compiler.pushNameScope(
          node->s(c10::Symbol::fromQualString("attr::name")).c_str());
    } else if (kind == symbols::poptorch::pop_name_scope) {
      _compiler.popNameScope();
    } else if (kind == symbols::poptorch::set_matmul_serialization) {
      poptorch::TensorId input = _value_map.tensor(node->input());
      _compiler.setMatMulSerialization(
          input, node->s(c10::Symbol::fromQualString("attr::mode")).c_str(),
          node->i(c10::Symbol::fromQualString("attr::factor")),
          node->i(c10::Symbol::fromQualString("attr::keep_precision")));
      _value_map.setTensor(node->output(), input);
    } else if (kind == symbols::poptorch::optimizer_group) {
      std::vector<poptorch::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       return _value_map.tensor(val);
                     });

      std::uint64_t group = node->i(c10::Symbol::fromQualString("attr::group"));
      _compiler.optimizerGroup(inputs, group);

    } else if (kind == symbols::poptorch::set_available_memory) {
      // Get the torch jit SSA for the input/output values.
      std::vector<poptorch::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       // Tuples aren't supported here but it's ok because
                       // we don't support any operations which actually take in
                       // tuples.
                       return _value_map.tensor(val);
                     });

      _compiler.setAvailableMemoryProportion(
          inputs, node->f(c10::Symbol::fromQualString(
                      "attr::availableMemoryProportion")));

      for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
        _value_map.setTensor(node->output(i), inputs[i]);
      }

    } else if (kind == c10::prim::Constant) {
      ERROR_ON_MSG(node->hasAttribute(c10::attr::value),
                   "Only None constants should be left in the graph after the "
                   "CanonicaliseConstants pass");
      _value_map.setTensor(node->output(), NoneTensor);
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
    } else if (kind == c10::prim::TupleUnpack ||
               kind == c10::prim::ListUnpack) {
      // Get the torch jit SSA for the input/output values.
      auto &tensors(_value_map.listTuple(node->input()));
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

      for (auto output : node->outputs()) {
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
    } else if (kind == symbols::poptorch::host_side_cast) {
      // Map to the input value since the type will be cast host side
      ERROR_ON_MSG(!_value_map.hasTensor(node->input()),
                   "Input to host side cast has not been registered");

      ERROR_ON_MSG(node->inputs().size() != 1,
                   "Host side cast should only have one input.");

      _value_map.setTensor(node->output(), _value_map.tensor(node->input()));

    } else if (kind == symbols::poptorch::multi_conv_part) {
      std::vector<poptorch::TensorId> inputs;
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
      std::vector<poptorch::TensorId> outputs = _compiler.endMultiConv();
      ERROR_ON_MSG(outputs.size() != node_outputs.size(),
                   "Wrong number of outputs for MultiConv. Expected "
                       << node_outputs.size() << " outputs but only received "
                       << outputs.size() << " outputs.");

      for (size_t i = 0; i < outputs.size(); i++) {
        _value_map.setTensor(node_outputs[i], outputs[i]);
      }

      logging::debug("{} was lowered to {} [{},{}]", nodeToString(node),
                     tensorNames(outputs[0], outputs.size()),
                     tensorTypesAndShapes(outputs[0], outputs.size()),
                     _compiler.getExecutionInfo().get());

    } else if (kind == symbols::poptorch::canonicalised_cpu_call) {
      // CPU callbacks are referenced by an string identifier.
      std::string id = node->s(c10::Symbol::fromQualString("attr::ID"));

      std::vector<poptorch::PopartType> input_types;
      std::vector<std::vector<std::size_t>> input_shapes;

      // Get the torch jit SSA for the input/output values.
      std::vector<poptorch::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       // Append type info from the inputs.
                       platformAgnosticTypeInfoFromIRType(val, &input_types,
                                                          &input_shapes);

                       return _value_map.tensor(val);
                     });

      std::vector<poptorch::PopartType> output_types;
      std::vector<std::vector<std::size_t>> output_shapes;

      for (torch::jit::Value *value : node->outputs()) {
        platformAgnosticTypeInfoFromIRType(value, &output_types,
                                           &output_shapes);
      }

      poptorch::TensorId first_output_tensor =
          _compiler.addCPUCallback(inputs, _callbacks[id], input_types,
                                   input_shapes, output_types, output_shapes);

      for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
        torch::jit::Value *output = node->output(i);
        poptorch::TensorId output_tensor = first_output_tensor + i;
        ERROR_ON_MSG(!_compiler.tensorIdIsValid(output_tensor),
                     "Output " << i << " doesn't exist of Node " << *node);
        _value_map.setTensor(output, output_tensor);
      }
    } else {
      ERROR("Couldn't find a registered operation for node");
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
  std::size_t num_input_tensors = graph_t_inputs.size() - _parameters.size();
  ERROR_ON(graph_t_inputs.size() != (in_tensors->size() + _parameters.size()));
  ERROR_ON(num_input_tensors + _parameters.size() != graph_t_inputs.size());

  // Store the number of inputs in _graph.inputs() before _parameters shrinks
  const size_t num_inputs = _graph.inputs().size() - _parameters.size();

  // Step 0, remove unused parameters
  // graph_t_inputs is updated but _graph.inputs() will retain unused parameters
  std::vector<bool> parameter_used(_parameters.size(), true);
  for (size_t index = 0; index < _parameters.size(); index++) {
    ERROR_ON(!parameter_used.at(index));
    auto value = graph_t_inputs[num_input_tensors + index];
    if (value->uses().empty()) {
      parameter_used.at(index) = false;

      logging::trace("Skipping unused parameter: {}",
                     _parameter_names.at(index));
    }
  }

  // Use remove-erase idiom to remove parameters with linear complexity
  maskVector(&_parameters, parameter_used);
  maskVector(&_parameter_names, parameter_used);
  maskVector(&graph_t_inputs, parameter_used, num_input_tensors);
  ERROR_ON(num_input_tensors + _parameters.size() != graph_t_inputs.size());

  // Step 1, add tensor inputs for all tensors in the hierachy and obtain the
  // the resulting popart IDs. This can be done with collapsed hierachy.
  // The collapsed hierachy has a 1 to 1 mapping with in_tensors, which contains
  // both "true" inputs and parameters.
  ValueMap::TensorList parameter_popart_ids;
  for (size_t index = 0; index < graph_t_inputs.size(); index++) {
    bool is_input_tensor = index < num_input_tensors;

    at::Tensor &tensor(is_input_tensor
                           ? (*in_tensors)[index]
                           : _parameters.at(index - num_input_tensors));

    std::vector<int64_t> dims = getTensorDimensions(tensor);

    if (index < num_input_tensors) {
      // Return the input tensor id for input tensor of given type and dims.
      auto id = _compiler.addInputTensor(
          typeToPopartStr(tensor.scalar_type()).c_str(), dims);
      _input_tensor_hooks.push_back(id);
    } else {
      ERROR_ON(graph_t_inputs[index]->uses().empty());

      const std::string &name = _parameter_names.at(index - num_input_tensors);
      std::string popart_type = typeToPopartStr(tensor.scalar_type());
      auto id = _compiler.addInitializedInputTensor(
          name.c_str(), popart_type.c_str(), dims, tensor.data_ptr());
      parameter_popart_ids.push_back(id);
    }
  }

  ERROR_ON(parameter_popart_ids.size() != _parameters.size());

  // Step 2, map the PopART tensor IDs to the JIT Value of the (not collapsed)
  // graph inputs
  logging::debug("graph(");
  auto input_tensor_it = _input_tensor_hooks.begin();
  size_t index = 0;
  for (torch::jit::Value *value : _graph.inputs()) {
    if (index == num_inputs) {
      // The rest are parameters
      break;
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
  ERROR_ON(parameter_popart_ids.size() != _parameters.size());

  // Step 3, map the PopART tensor IDs to the JIT Value of the parameters
  for (index = 0; index < parameter_popart_ids.size(); index++) {
    auto value = graph_t_inputs.at(num_input_tensors + index);
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

PopartConstant convertTensorConstantNode(const torch::jit::Node *node) {
  logging::LogContext ctx("convertTensorConstantNode processing " +
                          nodeToString(node));

  ERROR_ON_MSG(
      node->kind() != symbols::poptorch::tensor_constant,
      "Only a poptorch::tensor_constant can be converted into a popart "
      "constant");
  auto output_type =
      *node->output()->type()->expect<c10::TensorType>()->scalarType();
  auto tensor_type = node->t(c10::attr::value).scalar_type();

  ERROR_ON_MSG(output_type != tensor_type, "Output type is "
                                               << c10::toString(output_type)
                                               << " but tensor type is "
                                               << c10::toString(tensor_type));

  auto tensor = node->t(c10::attr::value);
  ERROR_ON(!tensor.is_contiguous());

  return {toPopartType(tensor.scalar_type()), tensor.data_ptr(),
          getTensorDimensions(tensor)};
}

HostSideConstant
convertHostSideTensorConstantNode(const torch::jit::Node *node) {
  logging::LogContext ctx("convertHostSideTensorConstantNode processing " +
                          nodeToString(node));
  ERROR_ON_MSG(node->kind() != symbols::poptorch::host_side_tensor_constant,
               "Only a poptorch::host_side_tensor_constant can be converted "
               "into a host side constant constant");

  auto tensor = node->t(c10::attr::value);
  ERROR_ON(!tensor.is_contiguous());

  return {toPopartType(tensor.scalar_type()), tensor.data_ptr(),
          tensor.nbytes(), getTensorDimensions(tensor)};
}

float pyFloatToFloatWithRangeCheck(const pybind11::handle &pyFloat) {
  // A python "float" is a double
  double value = pyFloat.cast<double>();
  ERROR_ON_MSG(value > std::numeric_limits<float>::max(),
               value << " is too high for a Popart float attribute.");
  ERROR_ON_MSG(value < std::numeric_limits<float>::lowest(),
               value << " is too low for a Popart float attribute.");
  return static_cast<float>(value);
}

void processListAttribute(
    const char *name,
    const std::shared_ptr<std::vector<PopartAttribute>> &attributes,
    const py::list &elements) {
  ERROR_ON(elements.empty());
  const auto &first_element = static_cast<py::object>(elements[0]);

  if (py::isinstance<py::int_>(first_element)) {
    std::vector<int64_t> ints;
    ints.reserve(elements.size());
    for (const auto &int_obj : elements) {
      ints.push_back(int_obj.cast<int64_t>());
    }
    attributes->emplace_back(name, ints);
    return;
  }

  if (py::isinstance<py::float_>(first_element)) {
    std::vector<float> floats;
    floats.reserve(elements.size());
    for (const auto &float_obj : elements) {
      floats.push_back(pyFloatToFloatWithRangeCheck(float_obj));
    }
    attributes->emplace_back(name, floats);
    return;
  }

  if (py::isinstance<py::str>(first_element)) {
    std::vector<std::unique_ptr<char[]>> strs;
    strs.reserve(elements.size());
    for (const auto &str : elements) {
      strs.emplace_back(stringToUniquePtr(str.cast<std::string>()));
    }
    attributes->emplace_back(name, strs);
    return;
  }

  ERROR("Invalid type for Popart attribute.");
}

std::shared_ptr<std::vector<PopartAttribute>>
convertCustomOpAttributes(const torch::jit::Node *node,
                          const py::function &attribute_accessor) {
  logging::LogContext ctx("convertCustomOpAttributes processing " +
                          nodeToString(node));
  std::string attributes_id_str(
      node->s(c10::Symbol::fromQualString("attr::attributes_id")));

  auto dict_obj = attribute_accessor(attributes_id_str);

  auto dict = dict_obj.cast<py::dict>();

  auto attributes = std::make_shared<std::vector<PopartAttribute>>();
  for (const auto &attribute : dict) {
    std::string name = attribute.first.cast<std::string>();

    if (py::isinstance<py::int_>(attribute.second)) {
      attributes->emplace_back(name.c_str(), attribute.second.cast<int64_t>());
    } else if (py::isinstance<py::float_>(attribute.second)) {
      attributes->emplace_back(name.c_str(),
                               pyFloatToFloatWithRangeCheck(attribute.second));
    } else if (py::isinstance<py::str>(attribute.second)) {
      attributes->emplace_back(
          name.c_str(),
          stringToUniquePtr(attribute.second.cast<std::string>()));
    } else if (py::isinstance<py::list>(attribute.second) ||
               py::isinstance<py::tuple>(attribute.second)) {
      processListAttribute(name.c_str(), attributes,
                           attribute.second.cast<py::tuple>());
    } else {
      ERROR("Invalid attribute type");
    }
  }

  return attributes;
}
} // namespace

LowerToPopartImpl::LowerToPopartImpl(
    torch::jit::Graph *g, std::vector<at::Tensor> params,
    std::vector<std::string> parameter_names,
    std::shared_ptr<InplaceOpHandler> inplace_op_handler, bool training,
    std::vector<Optimizer> &&opt, const SessionOptions &options,
    const py::function &attribute_accessor, CPUCallbackMap &&callback,
    const AnchorList &&anchors)
    : _graph(*g), _lowered(false), _parameters(std::move(params)),
      _parameter_names(std::move(parameter_names)),
      _inplace_op_handler(std::move(inplace_op_handler)), _optimizers(opt),
      _anchors(anchors), _compiler({training, options}), _callbacks(callback) {
  // Init the function implementation map. This map will be populated by
  // elements which look something like:
  /* {"popart::Foo", [&](const std::vector<poptorch::TensorId> &inputs,
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
#define BOOL i
#define STRING s
#define STRING_VEC ss

// Useful NOP macro
#define NONE

// The arguments are processed by extracting the given type using the above
// accessors, the name is converted into "attr::NAME" which is what pytorch JIT
// expects for attribute accessing.
#define ARG(Type, Name)                                                        \
  , convertType(node->Type(c10::Symbol::fromQualString("attr::" #Name)))

#define POPART_CONST_ARG(unused) , convertTensorConstantNode(node)
#define HOST_SIDE_CONST_ARG(unused)                                            \
  , std::move(convertHostSideTensorConstantNode(node))

#define POPART_ATTRIB_VEC_ARG(unused)                                          \
  , convertCustomOpAttributes(node, attribute_accessor)

#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(ns, symbolName, function, unused, Args, unused2)               \
  {symbols::ns::symbolName, [&](const std::vector<poptorch::TensorId> &inputs, \
                                torch::jit::Node *node) {                      \
     (void)(node);                                                             \
     return _compiler.function(inputs Args);                                   \
   }},

#define OP_DECL_NO_RETURN(ns, symbolName, function, unused, Args, unused2)     \
  {symbols::ns::symbolName, [&](const std::vector<poptorch::TensorId> &inputs, \
                                torch::jit::Node *node) {                      \
     _compiler.function(inputs Args);                                          \
     ERROR_ON_MSG(node->outputs().size() != 0,                                 \
                  "Void return function called on torch::jit::Node which has " \
                  "outputs");                                                  \
     return poptorch::TensorId{};                                              \
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
#undef STRING
#undef STRING_VEC
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC
  }; // End map initalizer.
}
} // namespace detail

LowerToPopart::LowerToPopart(
    torch::jit::Graph *graph, std::vector<at::Tensor> parameters,
    std::vector<std::string> parameter_names,
    const std::shared_ptr<InplaceOpHandler> &inplace_op_handler, bool training,
    std::vector<Optimizer> &&opt, const SessionOptions &options,
    const py::function &attribute_accessor, CPUCallbackMap callbacks,
    AnchorList &&anchors) {
  std::srand(std::time(nullptr));
  _impl = std::make_unique<detail::LowerToPopartImpl>(
      graph, std::move(parameters), std::move(parameter_names),
      inplace_op_handler, training, std::move(opt), std::move(options),
      attribute_accessor, std::move(callbacks), std::move(anchors));
}

void LowerToPopart::lower(std::vector<at::Tensor> *in_tensors) {
  _impl->lower(in_tensors);
}

std::shared_ptr<poptorch::PoplarExecutable> LowerToPopart::compile() {
  auto executable = _impl->compile();
  if (logging::outputPopartIR()) {
    logging::debug("Popart IR: {}", executable->getPopartIR());
  }
  return executable;
}

void LowerToPopart::compileAndExport(const std::string &output_filename) {
  // We cannot currently compile then export
  // so we need to decide now whether we want an executable
  // or to compile and export to file.
  _impl->compileAndExport(output_filename);
}
std::shared_ptr<poptorch::PoplarExecutable>
LowerToPopart::loadExecutableFromFile(const std::string &input_filename,
                                      std::int64_t offset) {
  return _impl->loadExecutableFromFile(input_filename, offset);
}
LowerToPopart::~LowerToPopart() = default;
LowerToPopart::LowerToPopart(LowerToPopart &&lower) {
  _impl = std::move(lower._impl);
}
} // namespace poptorch
