// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_SESSION_OPTIONS_PARSER_HPP
#define INCLUDE_POPTORCH_SESSION_OPTIONS_PARSER_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/CompilerTypes.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace popart_compiler {
class SessionOptions;
} // namespace popart_compiler

// Interface to parse a python object without adding a dependency on pybind
class IPyValue {
public:
  virtual std::function<void(int, int)> toFunction() const = 0;
  virtual bool isBoolean() const = 0;
  virtual bool toBoolean() const = 0;
  virtual bool isDouble() const = 0;
  virtual double toDouble() const = 0;
  virtual bool isInt() const = 0;
  virtual std::int64_t toInt64() const = 0;
  virtual std::uint64_t toUInt64() const = 0;
  virtual bool isString() const = 0;
  virtual std::string toString() const = 0;
  virtual bool isSetListOrTuple() const = 0;
  virtual void forEachInList(std::function<void(const IPyValue &)>) const = 0;
  virtual bool isDict() const = 0;
  virtual void forEachInDict(
      std::function<void(const IPyValue &, const IPyValue &)>) const = 0;
  // Return nullptr if the key doesn't exist
  virtual std::unique_ptr<IPyValue>
  getFromDict(const std::string &key) const = 0;
  // Return nullptr if index is out of bounds
  virtual std::unique_ptr<IPyValue> getFromList(std::uint64_t index) const = 0;
  virtual std::uint64_t getListSize() const = 0;
  virtual std::string type() const = 0;

  float toFloatWithRangeCheck() const;
  std::vector<std::string> toVectorString() const;
  virtual ~IPyValue() = default;
};

class SessionOptionsParser {
public:
  explicit SessionOptionsParser(const IPyValue &opts);
  popart_compiler::SessionOptions &options();
  ~SessionOptionsParser();

private:
  std::unique_ptr<popart_compiler::SessionOptions> _opts;
};

void processPrecisionOptions(const IPyValue &values_dict, bool dispatcher);

typedef std::function<std::unique_ptr<IPyValue>(const std::string &)>
    AttributeAccessor;

} // namespace poptorch

#endif // INCLUDE_POPTORCH_SESSION_OPTIONS_PARSER_HPP
