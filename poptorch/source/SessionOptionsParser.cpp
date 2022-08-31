// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poptorch/SessionOptionsParser.hpp"

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/Utils.hpp"
#include "poptorch/AutomaticCasting.hpp"
#include "poptorch/ImplicitCasting.hpp"
#include "poptorch_logging/Tracepoint.hpp"

namespace poptorch {

float IPyValue::toFloatWithRangeCheck() const {
  // A python "float" is a double
  double value = toDouble();

  ERROR_ON_MSG(value > std::numeric_limits<float>::max(),
               value << " is too high for a Popart float attribute.");
  ERROR_ON_MSG(value < std::numeric_limits<float>::lowest(),
               value << " is too low for a Popart float attribute.");
  return static_cast<float>(value);
}

std::vector<std::string> IPyValue::toVectorString() const {
  std::vector<std::string> out;
  out.reserve(getListSize());
  forEachInList([&out](const IPyValue &val) { out.push_back(val.toString()); });
  return out;
}

SessionOptionsParser::~SessionOptionsParser() = default;

popart_compiler::SessionOptions &SessionOptionsParser::options() {
  return *_opts;
}

SessionOptionsParser::SessionOptionsParser(const IPyValue &py_opts)
    : _opts(std::make_unique<popart_compiler::SessionOptions>()) {
  logging::LogContext ctx_func("parseSessionOptions");
  // steps, replicationFactor, profile
  auto &options = *_opts;

  py_opts.forEachInDict([&options, &py_opts](const IPyValue &name_val,
                                             const IPyValue &value) {
    const auto name = name_val.toString();
    logging::LogContext ctx("option: " + name);

    // Options excluded here:
    //  - patterns_level is handled at the same time as "patterns".
    //  - anchored_tensors is dealt with exclusively in Python.
    if (name == "patterns_level" || name == "anchored_tensors") {
      return;
    }

    if (name == "compilation_progress_bar_fn") {
      options.setCompilationProgressLogger(value.toFunction());
    } else if (value.isBoolean()) {
      options.addBoolOption(name.c_str(), value.toBoolean());
    } else if (value.isDouble()) {
      options.addDoubleOption(name.c_str(), value.toDouble());
    } else if (value.isInt()) {
      options.addUint64Option(name.c_str(), value.toUInt64());
    } else if (value.isString()) {
      options.addStringOption(name.c_str(), value.toString().c_str());
    } else if (value.isSetListOrTuple()) {
      value.forEachInList([&options, &name](const IPyValue &str_opt) {
        options.insertStringOption(name.c_str(), str_opt.toString().c_str());
      });
    } else if (value.isDict()) {
      if (name == "available_memory_proportion") {
        value.forEachInDict(
            [&options](const IPyValue &ipu, const IPyValue &memory) {
              options.setMemoryProportion(ipu.toUInt64(),
                                          memory.toFloatWithRangeCheck());
            });
      } else if (name == "patterns") {
        auto patterns_level = py_opts.getFromDict("patterns_level");
        ERROR_ON_MSG(patterns_level == nullptr,
                     "PopART option 'patterns' should not be set "
                     "without first setting 'patterns_level'.");

        options.setPatternsLevel(patterns_level->toUInt64());
        value.forEachInDict([&options](const IPyValue &pattern,
                                       const IPyValue &enabled) {
          options.addPattern(pattern.toString().c_str(), enabled.toBoolean());
        });
      } else if (name.rfind("location_", 0) == 0) {
        value.forEachInDict([&options, &name](const IPyValue &tensor,
                                              const IPyValue &location) {
          options.setTensorLocation(name.c_str(), tensor.toString().c_str(),
                                    location.toUInt64());
        });
      } else {
        value.forEachInDict([&options, &name](const IPyValue &str_key,
                                              const IPyValue &str_value) {
          options.insertStringPairOption(name.c_str(),
                                         str_key.toString().c_str(),
                                         str_value.toString().c_str());
        });
      }
    } else {
      ERROR("Unknown value type " << value.type() << " for option " << name);
    }
  });
}

/*
//TODO(T60710) Used in follow up patch
void processPrecisionOptions(const IPyValue &values_dict, bool dispatcher) {
  poptorch::setAutocastEnabled(
      values_dict.getFromDict("autocast_enabled")->toBoolean());

  auto policy = values_dict.getFromDict("autocast_policy_dict");
  std::vector<std::string> fp16;
  policy->getFromDict("fp16")->forEachInList(
      [&fp16](const IPyValue &val) { fp16.push_back(val.toString()); });
  setAutocastHalf(policy->getFromDict("fp16")->toVectorString());
  setAutocastFloat(policy->getFromDict("fp32")->toVectorString());
  setAutocastPromote(policy->getFromDict("promote")->toVectorString());
  setAutocastDemote(policy->getFromDict("demote")->toVectorString());

  auto hf_casting = static_cast<HalfFloatCasting>(
      values_dict.getFromDict("half_float_casting")->toInt64());

  ERROR_ON_MSG(
      dispatcher && hf_casting == HalfFloatCasting::FloatDowncastToHalf,
      "FloatDowncastToHalf is deprecated and not supported in the dispatcher");

  poptorch::setHalfFloatCastingBehavior(hf_casting);

  bool running_statistics_always_float;
  auto run_stats = values_dict.getFromDict("running_statistics_always_float");
  if (!run_stats) {
    running_statistics_always_float = !dispatcher;
  } else {
    running_statistics_always_float = run_stats->toBoolean();
  }
  ERROR_ON_MSG(dispatcher && running_statistics_always_float,
               "runningStatisticsAlwaysFloat is deprecated and not supported "
               "in the dispatcher. Simply cast the running statistics tensors "
               "to float at the pytorch level.");
  poptorch::setRunningStatisticsAlwaysFloat(running_statistics_always_float);
}
*/

void setPopartLogLevelUInt(std::uint64_t level) {
  ERROR_ON(level > static_cast<std::uint64_t>(logging::Level::Off) ||
           level == 5);
  popart_compiler::setPopartLogLevel(static_cast<logging::Level>(level));
}

void appendPoptorchMetadataToFile(const std::string &serialized_poptorch_data,
                                  const std::string &export_filename) {
  poptorch::logging::Tracepoint tp{__FUNCTION__};
  popart_compiler::Compiler::appendPoptorchMetadataToFile(
      serialized_poptorch_data.c_str(), serialized_poptorch_data.size(),
      export_filename.c_str());
}

bool mlirIsSupportedOnPlatform() { return POPTORCH_BUILD_MLIR_COMPILER == 1; }

} // namespace poptorch
