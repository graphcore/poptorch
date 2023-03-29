// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/CompilerTypes.hpp"
#include "poptorch_logging/LoggingLight.hpp"

namespace popart {
class any;
enum class DataType;
class ConstVoidData;
} // namespace popart

namespace poptorch {
namespace popart_compiler {

namespace detail {
struct CompilerImpl;
struct SessionOptionsImpl;
} // namespace detail

void throwTestError(TestErrorType type);

// Examines the supplied exception. If it is a popart or poplar exception,
// rethrow it as an ExceptionInfo subclass (which gives easy access to the
// exception detail)
void rethrowPopartOrPoplarException(const std::exception_ptr &eptr,
                                    const char *filename, uint64_t line);

void setPopartLogLevel(logging::Level level);

// Copies the value and constness of one parameter to another
void copyParam(Optimizer &dest_optim, const Optimizer &source_optim,
               const char *source, const char *dest);

class Compiler;
class SessionOptions {
public:
  SessionOptions();
  SessionOptions(SessionOptions &&);
  ~SessionOptions();
  // Disable copy: Move only
  SessionOptions(const SessionOptions &) = delete;
  SessionOptions &operator=(const SessionOptions &) = delete;

  void setMemoryProportion(std::uint32_t ipu, float memory);
  void setPatternsLevel(std::uint64_t level);
  void addPattern(const char *pattern, bool enabled);
  void setTensorLocation(const char *tensor, const char *option,
                         std::uint64_t value);
  void
  setCompilationProgressLogger(const std::function<void(int, int)> &logger);

  void addStringOption(const char *option, const char *value);
  void addUint64Option(const char *option, std::uint64_t value);
  void addBoolOption(const char *option, bool value);
  void addDoubleOption(const char *option, double value);
  // Insert a string option in an option container (set / list / vector)
  void insertStringOption(const char *option, const char *value);
  // Insert a key / value pair in an option map
  void insertStringPairOption(const char *option, const char *key,
                              const char *value);

  bool broadcastBuffers() const;
  bool hasInputReplication() const;

private:
  std::unique_ptr<detail::SessionOptionsImpl> _impl;
  friend Compiler;
};

// Represents an attribute used in a custom operation: popart uses popart::any
// to store the different values
class PopartAttribute {
public:
  // Templating works with g++ but not clang++
  PopartAttribute(const char *name, const int64_t &value);
  PopartAttribute(const char *name, const std::vector<int64_t> &values);
  PopartAttribute(const char *name, const float &value);
  PopartAttribute(const char *name, const std::vector<float> &values);
  PopartAttribute(const char *name, const std::unique_ptr<char[]> &str);
  PopartAttribute(const char *name,
                  const std::vector<std::unique_ptr<char[]>> &strs);

  // Required for opaque pointer
  PopartAttribute(PopartAttribute &&);
  PopartAttribute &operator=(PopartAttribute &&);
  ~PopartAttribute();

  popart::any *getValue();

  const char *name() const { return _name.get(); }

private:
  // Convert a "const char *" to a std::unique_ptr char*
  static std::unique_ptr<const char[]> cStrToUP(const char *name);

  // Use a pointer to circumvent the C++ ABI problems with std::string
  std::unique_ptr<const char[]> _name;

  // Use an opaque pointer to avoid the need for popart headers
  std::unique_ptr<popart::any> _any;
};

// A class to store all the data and info required to create a constant in the
// popart builder for convenience. Internally, it is a simple wrapper to
// popart::ConstVoidData.
class PopartConstant {
public:
  PopartConstant(const PopartType &popart_type, const void *data,
                 const std::vector<std::int64_t> &shape);

  ~PopartConstant(); // Required for opaque pointer

  const popart::ConstVoidData &getPopartData() const { return *_data; }

private:
  // Use an opaque pointer to avoid the need for popart headers
  std::unique_ptr<popart::ConstVoidData> _data;
};

// A class to store a constant which is simply returned, (possibly in a tuple
// or list) and is not inserted into Popart
class HostSideConstant {
public:
  HostSideConstant(const PopartType &popart_type, void *data, size_t data_size,
                   std::vector<std::int64_t> shape);

  PopartType popartType() const { return _popart_type; }

  const std::vector<std::int64_t> &shape() const { return _shape; }

  void copyDataTo(void *ptr) const;

private:
  const PopartType _popart_type;
  std::vector<uint8_t> _data;
  std::vector<std::int64_t> _shape;
};

class Compiler {
public:
  Compiler(bool is_training, const SessionOptions &options);
  ~Compiler();
  Compiler(Compiler &&compiler);

  TensorId addInputTensor(const char *type,
                          const std::vector<std::int64_t> &dims,
                          const char *overlap = "no_overlap");

  TensorId createTensorId(const char *name);

  void setCurrentPythonCodeLocation(const char *torch_node,
                                    const char *filename, std::uint64_t line,
                                    std::uint64_t col);

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<float>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define CHAR char
#define STRING const char *
#define STRING_VEC std::vector<const char *>
#define NONE
#define ARG(Type, Name) , Type Name
#define POPART_CONST_ARG(Name) , const PopartConstant &Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant &Name
#define POPART_ATTRIB_VEC_ARG(Name)                                            \
  , std::shared_ptr<std::vector<PopartAttribute>> Name
#define BODY_ARG(Name) NONE

// Create a function decl with the given call and arguments.
#define OP_DECL(Namespace, FuncName, function, OnnxImpl, Args, BodyArgs)       \
  TensorId function(const std::vector<TensorId> &inputs Args);

// Create a function decl with the given call and arguments which returns void.
#define OP_DECL_NO_RETURN(Namespace, FuncName, function, OnnxImpl, Args,       \
                          BodyArgs)                                            \
  void function(const std::vector<TensorId> &inputs Args);

#include "SupportedOperations.inc.hpp"

#undef OP_DECL
#undef OP_DECL_NO_RETURN
#undef BODY_ARG
#undef POPART_ATTRIB_VEC_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef ARG
#undef NONE
#undef STRING_VEC
#undef STRING
#undef CHAR
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC

  TensorId addInitializedInputTensor(const char *name, const char *type,
                                     const std::vector<std::int64_t> &dims,
                                     void *data);
  TensorId addInitializedInputTensor(const char *name, const char *type,
                                     const std::vector<std::int64_t> &dims,
                                     void *data, int comm_group_type,
                                     int shards, int variable_retrieval_mode);

  bool tensorIdIsValid(TensorId id) const;
  const char *tensorName(TensorId id) const;

  static const std::vector<std::int64_t> invalid_size;

  std::vector<std::int64_t> getSize(TensorId id) const;

  std::unique_ptr<char[]> getTensorDTypeString(TensorId id) const;

  bool isHostSideConstant(TensorId id) const;

  void addOutputType(OutputTypeShape type);

  // This function marks |output| as being read back from the device by the
  // host. |output_mode| determines how frequently that should happen.
  // clang-format off
  // "ALL":  Will return all popart batches.
  // "SUM": Will return the sum of all popart batches (I.E device iterations)
  // "EVERYN": Will return every N batch
  // "FINAL": Will return the last batch only
  // clang-format on
  void addOutputTensor(TensorId output,
                       PopartOutputMode output_mode = PopartOutputMode::N,
                       size_t output_return_period = 1,
                       const char *overlap = "no_overlap");

  void setUpInputOp(TensorId id, float *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(TensorId id, std::int32_t *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(TensorId id, bool *ptr,
                    const std::vector<std::int64_t> &dims);

  void setUpInputOp(TensorId id, std::int16_t *ptr,
                    const std::vector<std::int64_t> &dims,
                    bool float16 = false);

  // at::ScalarType::Byte
  void setUpInputOp(TensorId id, std::uint8_t *ptr,
                    const std::vector<std::int64_t> &dims);

  // at::ScalarType::Char
  void setUpInputOp(TensorId id, std::int8_t *ptr,
                    const std::vector<std::int64_t> &dims);

  // at::ScalarType::Byte
  void setUpOutputOp(TensorId id, std::uint8_t *ptr,
                     const std::vector<std::int64_t> &dims);

  // at::ScalarType::Char
  void setUpOutputOp(TensorId id, std::int8_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(TensorId id, float *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(TensorId id, std::int32_t *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(TensorId id, bool *ptr,
                     const std::vector<std::int64_t> &dims);

  void setUpOutputOp(TensorId id, std::int16_t *ptr,
                     const std::vector<std::int64_t> &dims);

  // Each std::set of tensors represents all the outputs of a node to set
  // the available memory proportion on. This function loops over the outer
  // vector, so the total number of nodes it will set the proportion on
  // will be inputs.size().
  void
  setAvailableMemoryProportion(const std::vector<std::set<TensorId>> &inputs,
                               float availableMemoryProportion);

  void setMatMulSerialization(TensorId matmul, const char *mode,
                              std::uint64_t factor,
                              std::uint64_t keep_precision);
  void clearActiveIpu();
  void setActiveIpu(std::uint64_t stage_id, std::int64_t phase_id,
                    std::int64_t ipu_id);

  void initSession(const std::vector<Optimizer> &opt,
                   const char *export_proto_filename);
  void setRngState(std::uint64_t seed,
                   const std::vector<std::uint32_t> &rng_state);

  std::vector<std::uint32_t> getRngState() const;
  std::uint64_t getRandomSeed() const;

  void saveExecutableToFile(const char *export_filename) const;
  void compileAndPrepareDevice();
  void loadEngineAndConnectStreams();
  void loadExecutableAndPrepareDevice(const char *import_filename);

  static void
  appendPoptorchMetadataToFile(const char *serialized_poptorch_metadata,
                               size_t metadata_length,
                               const char *export_filename);
  static std::vector<char>
  importPoptorchMetadataFromFile(const char *import_filename);

  TensorId addCPUCallback(const std::vector<TensorId> &inputs,
                          const CallbackMetadata &callback,
                          std::vector<PopartType> input_types,
                          std::vector<std::vector<std::size_t>> input_shapes,
                          std::vector<PopartType> output_types,
                          std::vector<std::vector<std::size_t>> output_shapes);

  void startSubgraph();

  TensorId endForLoop(std::int32_t trip_count, std::int64_t num_outputs,
                      const std::vector<TensorId> &inputs);

  void startIfBlock();

  void startElseBlock();

  TensorId endIfBlock(const TensorId &condition, std::size_t num_outputs);

  void pushNameScope(const char *name);

  void popNameScope();

  TensorId addUntypedInputTensor();
  // Write the weights into IPU memory from the pytorch tensor buffers in the
  // model.
  void copyWeightsToDevice(const std::vector<void *> &host_buffers);

  // Read the weights from IPU memory into the pytorch tensor buffers.
  void copyWeightsToHost(const std::vector<void *> &host_buffers);

  // Return the type of the given tensor.
  PopartType getPopartType(TensorId id) const;

  // Execute the compiled popart graph using poplar.
  void run();

  // Update the optimizers currently being run by the graph.
  void updateOptimizers(const std::vector<Optimizer> &optimizers);

  std::uint64_t batchPerStep() const;

  // Return the PopART batch dimensions [DeviceIterations * ReplicationFactor *
  // GradientAccumulation]
  std::uint64_t popartBatchDim() const;

  // Take the above and work out how much of it is being returned. ID must be
  // an anchor. The batch dim will be mutated depending on what the anchor is
  // returning.
  std::uint64_t popartBatchDimForAnchor(TensorId id) const;

  // Return a flat representation of the output types
  // For example: ( T0, T2, (T3, T4)) is represented as:
  // [ Tuple3, Tensor, Tensor, Tuple2, Tensor, Tensor ]
  const std::vector<OutputTypeShape> &outputTypes() const;

  // We return this as a unique char pointer to avoid leaking memory while
  // protecting the ABI boundry.
  std::unique_ptr<char[]> getPopartIR() const;

  // We return this as a unique char pointer to avoid leaking memory while
  // protecting the ABI boundry.
  std::set<std::unique_ptr<char[]>> getTensorNames() const;

  void optimizerGroup(const std::vector<TensorId> &inputs, int64_t group);

  std::vector<TensorMetadata> optimizerTensorMetadataList() const;

  void
  fillHostOptimizerStateTensorData(const std::vector<void *> &host_buffers);

  void
  writeDeviceOptimizerStateTensorData(const std::vector<void *> &host_buffers);

  std::unique_ptr<char[]> getExecutionInfo() const;

  void addMultiConvPart(const std::vector<TensorId> &inputs,
                        const std::vector<int64_t> &dilations,
                        const std::vector<int64_t> &kernel_shape,
                        const std::vector<int64_t> &pads,
                        const std::vector<int64_t> &strides);

  void setMultiConvAvailableMemoryProportions(const std::vector<double> &v);

  void setMultiConvPartialsTypes(const std::vector<int64_t> &partials_types);
  void
  setMultiConvEnableConvDithering(const std::vector<int64_t> &conv_dithering);

  void setMultiConvPlanType(int64_t plan_type);

  void setMultiConvPerConvReservedTiles(int64_t v);

  void setMultiConvCycleBackOff(double c);

  std::vector<TensorId> endMultiConv();

  void setAttribute(const char *attribute, const char *key, const char *value);
  void clearAttribute(const char *attribute, const char *key);

  void detachFromDevice();
  void attachToDevice();
  bool isAttachedToDevice() const;

  Timestamps getTimestamps() const;

  // Returns the number of cycles (on replica 0) run by the IPU for the last
  // model run.
  uint64_t getCycleCount() const;

  size_t getNumInputs() const;
  size_t getNumOutputs() const;

private:
  void assertTensorIs(PopartType dataType, TensorId id) const;

  // Make sure no overlap is specified for pipelined mode and that the output
  // mode is supported by PopART.
  void verifySettingsForOverlappedIO(PopartOutputMode output_mode);

  std::unique_ptr<detail::CompilerImpl> _impl;

  // Store the cycle account of last run, if the relevant option is enabled,
  // otherwise no_cycles
  int64_t _cycle_count;
  static constexpr int64_t no_cycles = -1;
  static constexpr const char *poptorch_opaque_name = "poptorch";
};

} // namespace popart_compiler
} // namespace poptorch
