// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/LowerToPopart.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <list>
#include <random>
#include <utility>

#include "PoptorchSymbols.hpp"
#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/PopartEnums.hpp"

#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

// Mapping between the SSA values of torch jit with the ssa values of popart.
// Each Value is either a single tensor or a tuple (Note: nested tuples are
// stored flattened).
class ValueMap {
public:
  using TensorList = std::vector<poptorch::TensorId>;

  poptorch::TensorId tensor(torch::jit::Value *value) const;
  const TensorList &tuple(torch::jit::Value *value) const;
  // Return the list of tensors without checking if it's a tuple or a single
  // tensor.
  const TensorList &tensors(torch::jit::Value *value) const;

  bool hasTensor(torch::jit::Value *value) const {
    return _map.count(value) == 1;
  }

  void setTensor(torch::jit::Value *value, poptorch::TensorId id);
  void setTuple(torch::jit::Value *value, const TensorList &tensors);

private:
  struct Data {
    explicit Data(poptorch::TensorId id) : is_tuple(false) {
      tensors.push_back(id);
    }
    explicit Data(TensorList tuple)
        : is_tuple(true), tensors(std::move(std::move(tuple))) {}
    bool is_tuple;
    TensorList tensors;
  };
  std::unordered_map<torch::jit::Value *, Data> _map;
};

poptorch::TensorId ValueMap::tensor(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  ERROR_ON_MSG(it->second.is_tuple, value->debugName() << " is not a tensor");
  ERROR_ON(it->second.tensors.size() != 1);
  return it->second.tensors.front();
}

const ValueMap::TensorList &ValueMap::tuple(torch::jit::Value *value) const {
  auto it = _map.find(value);
  ERROR_ON_MSG(it == _map.end(), value->debugName()
                                     << " not found in ValueMap");
  ERROR_ON_MSG(!it->second.is_tuple, value->debugName() << " is not a tuple");
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

void ValueMap::setTuple(torch::jit::Value *value,
                        const ValueMap::TensorList &tensors) {
  ERROR_ON_MSG(!_map.emplace(value, Data(tensors)).second,
               "Value " << value->debugName() << " already present in the map");
}

/*
 * Implementation of the lowering operation.
 */
class LowerToPopart {
public:
  LowerToPopart(torch::jit::Graph *g, std::vector<at::Tensor> *ins,
                std::vector<at::Tensor> params,
                std::vector<std::string> parameter_names, bool training,
                std::vector<Optimizer> &&opt, const SessionOptions &options);

  void lower();

  std::shared_ptr<poptorch::PoplarExecutable> compile();

private:
  torch::jit::Graph &_graph;

  std::vector<at::Tensor> &_in_tensors;

  std::vector<at::Tensor> _parameters;
  std::vector<std::string> _parameter_names;

  std::vector<poptorch::TensorId> _inputTensorHooks;

  std::vector<poptorch::TensorId> _outputTensorHooks;

  ValueMap _valueMap;

  // Optimizer from the user.
  const std::vector<Optimizer> _optimizers;

  using FunctionType = std::function<poptorch::TensorId(
      const std::vector<poptorch::TensorId> &inputs, torch::jit::Node *)>;
  std::unordered_map<c10::Symbol, FunctionType> _functionToImplementation;

  poptorch::Compiler _compiler;

  void lowerParameters();

  void lowerBody();

  void lowerReturn();

  std::string tensorNames(std::int64_t first_tensor, std::int64_t num_tensors);

  std::string tensorTypesAndShapes(std::int64_t first_tensor,
                                   std::int64_t num_tensors);
};

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

/*
 * Lower to popart impl.
 */
std::shared_ptr<poptorch::PoplarExecutable> LowerToPopart::compile() {
  logging::LogContext ctx("LowerToPopart::compiler ");
  // Init the session, this also involves compiling to poplar.
  _compiler.initSession(_optimizers);

  std::vector<at::ScalarType> data_types;
  for (auto id : _outputTensorHooks) {
    data_types.emplace_back(fromPopartType(_compiler.getPopartType(id)));
  }

  return std::make_shared<poptorch::PoplarExecutable>(
      std::move(_compiler), std::move(_inputTensorHooks),
      std::move(_outputTensorHooks), std::move(data_types), _parameter_names);
}

void LowerToPopart::lower() {
  // Lower the tensor parameters of the _graph to OpInputs.
  lowerParameters();

  // Lower the body of the _graph.
  lowerBody();

  lowerReturn();
}

void LowerToPopart::lowerReturn() {
  _compiler.addOutputType({OutputType::Type::Tuple,
                           static_cast<std::int64_t>(_graph.outputs().size())});
  // Recursively go through the output's type to
  // flatten its structure.
  // In the flat representation each 0 represent a
  // tensor, and each non zero value represent a
  // tuple of that size.
  // e.g: (T0, T1, (T2, T3), T4)
  // [ 4, 0, 0, 2, 0, 0, 0 ]
  std::function<void(c10::TypePtr)> process_type;
  process_type = [this, &process_type](const c10::TypePtr &type) {
    switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      _compiler.addOutputType({OutputType::Type::Tensor});
      break;
    }
    case c10::TypeKind::TupleType: {
      auto tuple = type->expect<c10::TupleType>();
      _compiler.addOutputType(
          {OutputType::Type::Tuple,
           static_cast<std::int64_t>(tuple->elements().size())});
      for (const auto &elt_type : tuple->elements()) {
        process_type(elt_type);
      }
      break;
    }
    default:
      ERROR("Unsupported output type '" << c10::typeKindToString(type->kind()));
    }
  };
  for (torch::jit::Value *value : _graph.outputs()) {
    if (value->type()->kind() == c10::TypeKind::ListType) {
      c10::TypeKind elt_kind =
          value->type()->expect<c10::ListType>()->getElementType()->kind();
      ERROR_ON_MSG(elt_kind != c10::TypeKind::TensorType,
                   "Unsupported list type " << c10::typeKindToString(elt_kind));
      std::int64_t num_tensors =
          static_cast<std::int64_t>(_valueMap.tensors(value).size());
      _compiler.addOutputType({OutputType::Type::List, num_tensors});
      for (std::int64_t i = 0; i < num_tensors; ++i) {
        _compiler.addOutputType({OutputType::Type::Tensor});
      }
    } else {
      process_type(value->type());
    }
    for (auto id : _valueMap.tensors(value)) {
      _compiler.addOutputTensor(id);
      _outputTensorHooks.push_back(id);
    }
  }
}

std::string LowerToPopart::tensorNames(std::int64_t first_tensor,
                                       std::int64_t num_tensors) {
  std::string sep{};
  std::string names;
  for (std::int64_t i = 0; i < num_tensors; ++i) {
    names += sep + _compiler.tensorName(first_tensor + i);
    sep = ", ";
  }
  return names;
}

std::string LowerToPopart::tensorTypesAndShapes(std::int64_t first_tensor,
                                                std::int64_t num_tensors) {
  std::string sep{};
  std::string shapes;

  const char *shape_inf_failed = "(shape inference failed)";

  for (std::int64_t i = 0; i < num_tensors; ++i) {
    std::ostringstream shape_str;

    try {
      auto tensor_shape = _compiler.getSize(first_tensor + i);

      auto dtype_chars = _compiler.getTensorDTypeString(first_tensor + i);
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
void LowerToPopart::lowerBody() {
  logging::debug("Graph lowered to Popart {");
  for (torch::jit::Node *node : _graph.nodes()) {
    logging::LogContext ctx("LowerToPopart::lowerBody Processing " +
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
                       return _valueMap.tensor(val);
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
        _valueMap.setTensor(output, output_tensor);
      }

      if (!_compiler.isHostSideConstant(first_output_tensor)) {
        logging::debug(
            "{} was lowered to {} [{},{}]", nodeToString(node),
            tensorNames(first_output_tensor, node->outputs().size()),
            tensorTypesAndShapes(first_output_tensor, node->outputs().size()),
            _compiler.getExecutionInfo().get());
      }
    } else if (kind == symbols::poptorch::end_ipu_block) {
      _compiler.clearActiveIpu();
    } else if (kind == symbols::poptorch::begin_ipu_block) {
      _compiler.setActiveIpu(
          node->i(c10::Symbol::fromQualString("attr::stage")),
          node->i(c10::Symbol::fromQualString("attr::phase")),
          node->i(c10::Symbol::fromQualString("attr::ipu")));
    } else if (kind == symbols::poptorch::set_matmul_serialization) {
      poptorch::TensorId input = _valueMap.tensor(node->input());
      _compiler.setMatMulSerialization(
          input, node->s(c10::Symbol::fromQualString("attr::mode")).c_str(),
          node->i(c10::Symbol::fromQualString("attr::factor")),
          node->i(c10::Symbol::fromQualString("attr::keep_precision")));
      _valueMap.setTensor(node->output(), input);
    } else if (kind == symbols::poptorch::optimizer_group) {
      std::vector<poptorch::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       return _valueMap.tensor(val);
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
                       return _valueMap.tensor(val);
                     });

      _compiler.setAvailableMemoryProportion(
          inputs, node->f(c10::Symbol::fromQualString(
                      "attr::availableMemoryProportion")));

      for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
        _valueMap.setTensor(node->output(i), inputs[i]);
      }

    } else if (kind == c10::prim::Constant) {
      ERROR_ON_MSG(node->hasAttribute(c10::attr::value),
                   "Only None constants should be left in the graph after the "
                   "CanonicaliseConstants pass");
      _valueMap.setTensor(node->output(), NoneTensor);
    } else if (kind == c10::prim::TupleConstruct ||
               kind == c10::prim::ListConstruct) {
      // Get the torch jit SSA for the input/output values.
      torch::jit::Value *output = node->output();

      // Add the values to the value map.
      ValueMap::TensorList tuple;
      for (torch::jit::Value *ids : node->inputs()) {
        for (auto tensor : _valueMap.tensors(ids)) {
          tuple.push_back(tensor);
        }
      }
      _valueMap.setTuple(output, tuple);
    } else if (kind == c10::prim::TupleUnpack ||
               kind == c10::prim::ListUnpack) {
      // Get the torch jit SSA for the input/output values.
      auto tensors = _valueMap.tuple(node->input());
      auto tensor_it = tensors.begin();
      std::function<void(c10::TypePtr, ValueMap::TensorList &)> process_output;

      // Find out how many tensors a given output consumes by walking
      // recursively through its type.
      process_output = [&](const c10::TypePtr &type,
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
            process_output(elt_type, tensorList);
          }
          break;
        }
        default:
          ERROR("Unsupported type '" << c10::typeKindToString(type->kind()));
        }
      };
      for (auto output : node->outputs()) {
        ValueMap::TensorList tensor_list;
        process_output(output->type(), tensor_list);
        switch (output->type()->kind()) {
        case c10::TypeKind::TensorType: {
          ERROR_ON(tensor_list.size() != 1);
          _valueMap.setTensor(output, tensor_list.front());
          break;
        }
        case c10::TypeKind::TupleType: {
          _valueMap.setTuple(output, tensor_list);
          break;
        }
        default:
          ERROR("Unsupported parameter type '"
                << c10::typeKindToString(output->type()->kind()));
        }
      }
      ERROR_ON_MSG(tensor_it != tensors.end(), "Didn't unpack all the tensors");
    } else if (kind == symbols::poptorch::host_side_cast) {
      // Map to the input value since the type will be casted host side
      ERROR_ON_MSG(!_valueMap.hasTensor(node->input()),
                   "Input to host side cast has not been registered");

      ERROR_ON_MSG(node->inputs().size() != 1,
                   "Host side cast should only have one input.");

      _valueMap.setTensor(node->output(), _valueMap.tensor(node->input()));

    } else {
      ERROR("Couldn't find a registered operation for node");
    }
  }
  logging::debug("}");
}

void LowerToPopart::lowerParameters() {
  std::size_t num_inputs =
      _graph.param_node()->outputs().size() - _parameters.size();
  std::size_t index = 0;
  auto tensor_it = _in_tensors.begin();

  std::function<void(c10::TypePtr, ValueMap::TensorList &)> process_input;
  process_input = [&](const c10::TypePtr &type,
                      ValueMap::TensorList &tensorList) {
    switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      ERROR_ON(tensor_it == _in_tensors.end());
      auto tensor = *tensor_it;
      tensor_it++;
      // Convert the tensor type to the correct vector size.
      std::vector<int64_t> dims = getTensorDimensions(tensor);

      // Return the input tensor id for input tensor of given type and dims.
      poptorch::TensorId id = _compiler.addInputTensor(
          typeToPopartStr(tensor.scalar_type()).c_str(), dims);

      // Record the id so we can map back to the pytorch tensor.
      tensorList.push_back(id);
      _inputTensorHooks.push_back(id);
      break;
    }
    case c10::TypeKind::TupleType: {
      auto tuple = type->expect<c10::TupleType>();
      for (const auto &elt_type : tuple->elements()) {
        process_input(elt_type, tensorList);
      }
      break;
    }
    default:
      ERROR("Unsupported parameter type '"
            << c10::typeKindToString(type->kind()) << "' for input " << index);
    }
  };

  for (torch::jit::Value *value : _graph.inputs()) {
    if (index < num_inputs) {
      // Lower user provided input
      ERROR_ON(value->node()->kind() != c10::prim::Param);
      ValueMap::TensorList tensors;
      process_input(value->type(), tensors);
      switch (value->type()->kind()) {
      case c10::TypeKind::TensorType: {
        ERROR_ON(tensors.size() != 1);
        _valueMap.setTensor(value, tensors.front());
        break;
      }
      case c10::TypeKind::TupleType: {
        _valueMap.setTuple(value, tensors);
        break;
      }
      default:
        ERROR("Unsupported parameter type '"
              << c10::typeKindToString(value->type()->kind()) << "' for input "
              << index);
      }
    } else if (!value->uses().empty()) {
      ERROR_ON_MSG(tensor_it != _in_tensors.end(),
                   "Not all the input tensors have been used");
      // Lower the other params (i.e the weights)
      at::Tensor &tensor_as_param = _parameters[index - num_inputs];
      const std::string &name = _parameter_names.at(index - num_inputs);

      // Convert the tensor type to the correct vector size.
      std::vector<int64_t> dims = getTensorDimensions(tensor_as_param);

      // Unpack the elem type into its Popart type.
      std::string popart_type = typeToPopartStr(tensor_as_param.scalar_type());
      _valueMap.setTensor(value, _compiler.addInitializedInputTensor(
                                     name.c_str(), popart_type.c_str(), dims,
                                     tensor_as_param.data_ptr()));
    } else {
      logging::trace("Skipping unused parameter: {}",
                     _parameter_names.at(index - num_inputs));

      size_t erase_at = index - num_inputs;
      _parameters.erase(_parameters.begin() + erase_at);
      _parameter_names.erase(_parameter_names.begin() + erase_at);
      --index;
    }
    ++index;
  }
}

// Helper to let us filter string arguments into const char*s. This is to catch
// the std::string produced by some attributes before they cross the ABI
// boundary.
namespace {

// Default template conversion, just return the type.
template <typename T> struct StringConvertorHelper {
  explicit StringConvertorHelper(T x) : value(std::move(x)) {}
  T value;

  operator T() { return value; } // NOLINT
};

// String, return const char*.
template <> struct StringConvertorHelper<std::string> {
  explicit StringConvertorHelper(const std::string &x) : value(x) {}
  const std::string &value;

  operator const char *() { return value.c_str(); } // NOLINT
};

// Function to create the conversion helper. To allow template type deduction
// and template specialization at the same time.
template <typename T> StringConvertorHelper<T> convertString(T t) {
  return StringConvertorHelper<T>{t};
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

} // namespace

LowerToPopart::LowerToPopart(torch::jit::Graph *g, std::vector<at::Tensor> *ins,
                             std::vector<at::Tensor> params,
                             std::vector<std::string> parameter_names,
                             bool training, std::vector<Optimizer> &&opt,
                             const SessionOptions &options)
    : _graph(*g), _in_tensors(*ins), _parameters(std::move(params)),
      _parameter_names(std::move(parameter_names)), _optimizers(opt),
      _compiler({training, options}) {
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

// Useful NOP macro
#define NONE

// The arguments are processed by extracting the given type using the above
// accessors, the name is converted into "attr::NAME" which is what pytorch JIT
// expects for attribute accessing.
#define ARG(Type, Name)                                                        \
  , convertString(node->Type(c10::Symbol::fromQualString("attr::" #Name)))

#define POPART_CONST_ARG(unused) , convertTensorConstantNode(node)
#define HOST_SIDE_CONST_ARG(unused)                                            \
  , std::move(convertHostSideTensorConstantNode(node))

#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(ns, symbolName, function, unused, Args, unused2)               \
  {symbols::ns::symbolName, [&](const std::vector<poptorch::TensorId> &inputs, \
                                torch::jit::Node *node) {                      \
     (void)(node);                                                             \
     return _compiler.function(inputs Args);                                   \
   }},

#include "popart_compiler/SupportedOperations.inc.hpp"

#undef BODY_STR_ARG
#undef STR_ARG
#undef BODY_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC
  }; // End map initalizer.
}
} // namespace

std::shared_ptr<poptorch::PoplarExecutable>
lowerToPopart(torch::jit::Graph *graph, std::vector<at::Tensor> *in_tensors,
              std::vector<at::Tensor> parameters,
              std::vector<std::string> parameter_names, bool training,
              std::vector<Optimizer> &&opt, const SessionOptions &options) {
  std::srand(std::time(nullptr));

  LowerToPopart lower_impl{graph,
                           in_tensors,
                           std::move(parameters),
                           std::move(parameter_names),
                           training,
                           std::move(opt),
                           std::move(options)};
  lower_impl.lower();

  auto executable = lower_impl.compile();
  if (logging::outputPopartIR()) {
    logging::debug("Popart IR: {}", executable->getPopartIR());
  }
  return executable;
}

} // namespace poptorch
