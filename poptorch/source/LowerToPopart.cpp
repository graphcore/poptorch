// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/LowerToPopart.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <list>
#include <random>
#include <utility>

#include "PoptorchSymbols.h"
#include "popart_compiler/Compiler.hpp"
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

  poptorch::TensorId Tensor(torch::jit::Value *value) const;
  const TensorList &Tuple(torch::jit::Value *value) const;
  bool IsTuple(torch::jit::Value *value) const;

  void SetTensor(torch::jit::Value *value, poptorch::TensorId id);
  void SetTuple(torch::jit::Value *value, const TensorList &tuple);

private:
  struct Data {
    explicit Data(poptorch::TensorId id) : isTuple(false) {
      tensors.push_back(id);
    }
    explicit Data(TensorList tuple) : isTuple(true), tensors(tuple) {}
    bool isTuple;
    TensorList tensors;
  };
  std::unordered_map<torch::jit::Value *, Data> map;
};

bool ValueMap::IsTuple(torch::jit::Value *value) const {
  auto it = map.find(value);
  ERROR_ON_MSG(it == map.end(), value->debugName() << " not found in ValueMap");
  return it->second.isTuple;
}

poptorch::TensorId ValueMap::Tensor(torch::jit::Value *value) const {
  auto it = map.find(value);
  ERROR_ON_MSG(it == map.end(), value->debugName() << " not found in ValueMap");
  ERROR_ON_MSG(it->second.isTuple, value->debugName() << " is not a tensor");
  ERROR_ON(it->second.tensors.size() != 1);
  return it->second.tensors.front();
}

const ValueMap::TensorList &ValueMap::Tuple(torch::jit::Value *value) const {
  auto it = map.find(value);
  ERROR_ON_MSG(it == map.end(), value->debugName() << " not found in ValueMap");
  ERROR_ON_MSG(!it->second.isTuple, value->debugName() << " is not a tuple");
  return it->second.tensors;
}

void ValueMap::SetTensor(torch::jit::Value *value, poptorch::TensorId id) {
  ERROR_ON_MSG(!map.emplace(value, Data(id)).second,
               "Value " << value->debugName() << " already present in the map");
}

void ValueMap::SetTuple(torch::jit::Value *value,
                        const ValueMap::TensorList &tensors) {
  ERROR_ON_MSG(!map.emplace(value, Data(tensors)).second,
               "Value " << value->debugName() << " already present in the map");
}

/*
 * Implementation of the lowering operation.
 */
class LowerToPopart {
public:
  LowerToPopart(
      torch::jit::Graph &g, std::vector<at::Tensor> &ins,
      std::vector<at::Tensor> &params, std::uint64_t steps, bool training,
      std::uint64_t replicationFactor, std::uint64_t gradientAccumulation,
      const std::unordered_map<std::string, std::pair<float, bool>> &opt,
      bool profile_);

  void Lower();

  std::shared_ptr<poptorch::PoplarExecutable> Compile();

private:
  torch::jit::Graph &graph;

  std::vector<at::Tensor> &inTensors;

  std::vector<at::Tensor> &parameters;

  std::vector<poptorch::TensorId> inputTensorHooks;

  std::vector<poptorch::TensorId> outputTensorHooks;

  ValueMap valueMap;

  // Optimizer from the user.
  const std::unordered_map<std::string, std::pair<float, bool>> &optimizer;

  using FunctionType = std::function<poptorch::TensorId(
      const std::vector<poptorch::TensorId> &inputs, torch::jit::Node *)>;
  std::unordered_map<c10::Symbol, FunctionType> functionToImplementation;

  poptorch::Compiler compiler;

  bool profile;

  void LowerParameters();

  void LowerBody();

  void LowerReturn();
};

/*
 * Static helper functions.
 */

std::string typeToPopart(at::ScalarType type) {
  if (type == at::ScalarType::Float) {
    return "FLOAT";
  } else if (type == at::ScalarType::Int || type == at::ScalarType::Long) {
    return "INT32";
  }

  logging::err("Unimplemented type '{}'", type);
  return "UNIMPLEMENTED";
}

std::vector<int64_t> GetTensorDimensions(const at::Tensor &tensor) {
  std::vector<int64_t> dims;
  std::transform(tensor.sizes().begin(), tensor.sizes().end(),
                 std::back_inserter(dims), [](std::int64_t i) { return i; });
  return dims;
}

/*
 * Lower to popart impl.
 */
std::shared_ptr<poptorch::PoplarExecutable> LowerToPopart::Compile() {
  // Init the session, this also involves compiling to poplar.
  compiler.InitSession(profile, optimizer);

  return std::make_shared<poptorch::PoplarExecutable>(
      std::move(compiler), std::move(inputTensorHooks),
      std::move(outputTensorHooks), profile);
}

void LowerToPopart::Lower() {
  // Lower the tensor parameters of the graph to OpInputs.
  LowerParameters();

  // Lower the body of the graph.
  LowerBody();

  LowerReturn();
}

void LowerToPopart::LowerReturn() {
  for (torch::jit::Value *value : graph.outputs()) {
    if (valueMap.IsTuple(value)) {
      for (auto id : valueMap.Tuple(value)) {
        compiler.AddOutput(id);
        outputTensorHooks.push_back(id);
      }
    } else {
      auto id = valueMap.Tensor(value);
      compiler.AddOutput(id);
      outputTensorHooks.push_back(id);
    }
  }
}

// Lower the main body of the graph.
void LowerToPopart::LowerBody() {
  for (torch::jit::Node *node : graph.nodes()) {
    // Switch/lookup based on the actual int value.
    const c10::Symbol kind = node->kind();

    auto itr = functionToImplementation.find(kind);
    if (itr != functionToImplementation.end()) {
      // Get the torch jit SSA for the input/output values.
      std::vector<poptorch::TensorId> inputs;
      std::transform(node->inputs().begin(), node->inputs().end(),
                     std::back_inserter(inputs), [&](torch::jit::Value *val) {
                       // Tuples aren't supported here but it's ok because
                       // we don't support any operations which actually take in
                       // tuples.
                       return valueMap.Tensor(val);
                     });

      torch::jit::Value *output = node->output();
      // Call the callback.
      valueMap.SetTensor(output, itr->second(inputs, node));
    } else if (kind == Symbols::poptorch::begin_ipu_block) {
      compiler.SetActiveIpu(node->i(c10::Symbol::fromQualString("attr::ipu")));
    } else if (kind == Symbols::poptorch::end_ipu_block) {
      // NOP for now.
    } else if (kind == c10::prim::TupleConstruct ||
               kind == c10::prim::ListConstruct) {
      // Get the torch jit SSA for the input/output values.
      torch::jit::Value *output = node->output();

      // Add the values to the value map.
      ValueMap::TensorList tuple;
      for (torch::jit::Value *ids : node->inputs()) {
        tuple.push_back(valueMap.Tensor(ids));
      }
      valueMap.SetTuple(output, tuple);
    } else if (kind == c10::prim::TupleUnpack ||
               kind == c10::prim::ListUnpack) {
      // Get the torch jit SSA for the input/output values.
      auto tensors = valueMap.Tuple(node->input());
      auto tensorIt = tensors.begin();
      std::function<void(c10::TypePtr, ValueMap::TensorList &)> processOutput;

      // Find out how many tensors a given output consumes by walking
      // recursively through its type.
      processOutput = [&](c10::TypePtr type, ValueMap::TensorList &tensorList) {
        switch (type->kind()) {
        case c10::TypeKind::TensorType: {
          ERROR_ON_MSG(tensorIt == tensors.end(),
                       "Not enough tensors to unpack");
          tensorList.push_back(*tensorIt);
          tensorIt++;
          break;
        }
        case c10::TypeKind::TupleType: {
          auto tuple = type->expect<c10::TupleType>();
          for (auto eltType : tuple->elements()) {
            processOutput(eltType, tensorList);
          }
          break;
        }
        default:
          ERROR("Unsupported type '" << c10::typeKindToString(type->kind()));
        }
      };
      for (auto output : node->outputs()) {
        ValueMap::TensorList tensorList;
        processOutput(output->type(), tensorList);
        switch (output->type()->kind()) {
        case c10::TypeKind::TensorType: {
          ERROR_ON(tensorList.size() != 1);
          valueMap.SetTensor(output, tensorList.front());
          break;
        }
        case c10::TypeKind::TupleType: {
          valueMap.SetTuple(output, tensorList);
          break;
        }
        default:
          ERROR("Unsupported parameter type '"
                << c10::typeKindToString(output->type()->kind()));
        }
      }
      ERROR_ON_MSG(tensorIt != tensors.end(), "Didn't unpack all the tensors");
    } else {
      logging::err("Couldn't find a registered operation for node {}", *node);
    }
  }
}

void LowerToPopart::LowerParameters() {
  std::size_t numInputs =
      graph.param_node()->outputs().size() - parameters.size();
  std::size_t index = 0;
  auto tensorIt = inTensors.begin();

  std::function<void(c10::TypePtr, ValueMap::TensorList &)> processInput;
  processInput = [&](c10::TypePtr type, ValueMap::TensorList &tensorList) {
    switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      ERROR_ON(tensorIt == inTensors.end());
      auto tensor = *tensorIt;
      tensorIt++;
      // Convert the tensor type to the correct vector size.
      std::vector<int64_t> dims = GetTensorDimensions(tensor);

      // Return the input tensor id for input tensor of given type and dims.
      poptorch::TensorId id = compiler.AddInputTensor(
          typeToPopart(tensor.scalar_type()).c_str(), dims);

      // Record the id so we can map back to the pytorch tensor.
      tensorList.push_back(id);
      inputTensorHooks.push_back(id);
      break;
    }
    case c10::TypeKind::TupleType: {
      auto tuple = type->expect<c10::TupleType>();
      for (auto eltType : tuple->elements()) {
        processInput(eltType, tensorList);
      }
      break;
    }
    default:
      ERROR("Unsupported parameter type '"
            << c10::typeKindToString(type->kind()) << "' for input " << index);
    }
  };

  for (torch::jit::Value *value : graph.param_node()->outputs()) {
    if (index < numInputs) {
      // Lower user provided input
      ERROR_ON(value->node()->kind() != c10::prim::Param);
      ValueMap::TensorList tensors;
      processInput(value->type(), tensors);
      switch (value->type()->kind()) {
      case c10::TypeKind::TensorType: {
        ERROR_ON(tensors.size() != 1);
        valueMap.SetTensor(value, tensors.front());
        break;
      }
      case c10::TypeKind::TupleType: {
        valueMap.SetTuple(value, tensors);
        break;
      }
      default:
        ERROR("Unsupported parameter type '"
              << c10::typeKindToString(value->type()->kind()) << "' for input "
              << index);
      }
    } else {
      ERROR_ON_MSG(tensorIt != inTensors.end(),
                   "Not all the input tensors have been used");
      // Lower the other params (i.e the weights)
      at::Tensor &tensorAsParam = parameters[index - numInputs];

      // Convert the tensor type to the correct vector size.
      std::vector<int64_t> dims = GetTensorDimensions(tensorAsParam);

      // Unpack the elem type into its Popart type.
      std::string popartType = typeToPopart(tensorAsParam.scalar_type());

      valueMap.SetTensor(value, compiler.AddInitializedInputTensor(
                                    "Weight", popartType.c_str(), dims,
                                    tensorAsParam.data_ptr()));
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
  explicit StringConvertorHelper(T x) : value(x) {}
  T value;

  operator T() { return value; }
};

// String, return const char*.
template <> struct StringConvertorHelper<std::string> {
  explicit StringConvertorHelper(std::string &x) : value(x) {}
  std::string &value;

  operator const char *() { return value.c_str(); }
};

// Function to create the conversion helper. To allow template type deduction
// and template specialization at the same time.
template <typename T> StringConvertorHelper<T> convertString(T t) {
  return StringConvertorHelper<T>{t};
}

} // namespace

LowerToPopart::LowerToPopart(
    torch::jit::Graph &g, std::vector<at::Tensor> &ins,
    std::vector<at::Tensor> &params, std::uint64_t steps, bool training,
    std::uint64_t replicationFactor, std::uint64_t gradientAccumulation,
    const std::unordered_map<std::string, std::pair<float, bool>> &opt,
    bool profile_)
    : graph(g), inTensors(ins), parameters(params), optimizer(opt),
      compiler({training, steps, replicationFactor, gradientAccumulation}),
      profile(profile_) {
  // Init the function implementation map. This map will be populated by
  // elements which look something like:
  /* {"popart::Foo", [&](const std::vector<poptorch::TensorId> &inputs,
     torch::jit::Node *node) { return compiler.foo(inputs,
          node->i("attr::SomeIntegerAttr"),
    node->i("attr::SomeOtherIntegerAttr"), node->is("attr::AnIntArrayAttr"),
    node->f("attr::AFloatAttr"));
      }
    },
  */
  // Essentially this is just a map from the string IR symbol to a function to
  // be called that implements it. Those functions are also autogenerated by the
  // same macros in compiler.hpp and compiler.cpp.
  functionToImplementation = {
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
#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(ns, funcName, function, unused, Args, unused2)                 \
  {Symbols::ns::funcName, [&](const std::vector<poptorch::TensorId> &inputs,   \
                              torch::jit::Node *node) {                        \
     (void)(node);                                                             \
     return compiler.function(inputs Args);                                    \
   }},

#include "popart_compiler/SupportedOperations.inc.h"

#undef BODY_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef INT_VEC
#undef FLOAT_VEC
#undef FLOAT
#undef INT
#undef BOOL
#undef STRING
  }; // End map initalizer.
}

} // namespace

std::shared_ptr<poptorch::PoplarExecutable> lowerToPopart(
    torch::jit::Graph &graph, std::vector<at::Tensor> &inTensors,
    std::vector<at::Tensor> &parameters, std::uint64_t steps, bool training,
    std::uint64_t replicationFactor, std::uint64_t gradientAccumulation,
    const std::unordered_map<std::string, std::pair<float, bool>> &opt,
    bool profile) {
  std::srand(std::time(nullptr));

  LowerToPopart lower_impl{
      graph,    inTensors,         parameters,           steps,
      training, replicationFactor, gradientAccumulation, opt,
      profile};
  lower_impl.Lower();

  return lower_impl.Compile();
}

} // namespace poptorch
