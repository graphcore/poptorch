// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "IDispatch.hpp"

#include <memory>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../CommonHelperFunctions.hpp"
#include "../Tensor.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {

IDispatch::IDispatch(TensorStore *tensor_store) {
  ERROR_ON(tensor_store == nullptr);
  _tensor_store = tensor_store;
}

void IDispatch::setPythonStack(
    const std::vector<torch::jit::StackEntry> &stack) {
  setCurrentCodeLocation(getPythonInterpreterSourceRange(stack));
}

void *IDispatch::getDataSource(torch::jit::Value *value) {
  auto *record = _mapper.rawTensorRecord(value);
  if (record == nullptr) {
    logging::trace("JIT value not tracked {}", reinterpret_cast<void *>(value));
    return nullptr;
  }
  return record->tensor_details->host_buffer->data();
}

bool IDispatch::isParameter(torch::jit::Value *value) {
  auto *record = _mapper.rawTensorRecord(value);
  ERROR_ON_MSG(record == nullptr,
               "JIT value not tracked " << reinterpret_cast<void *>(value));
  return record->tensor_details->is_parameter;
}

void IDispatch::setParameterName(const at::Tensor &tensor,
                                 const std::string &name) {
  _mapper.setParameterName(tensor, name);
}

std::string IDispatch::getParameterName(torch::jit::Value *value) {
  auto *record = _mapper.rawTensorRecord(value);
  if (record == nullptr) {
    logging::trace("JIT value not tracked {}", reinterpret_cast<void *>(value));
    return "";
  }
  ERROR_ON_MSG(!record->tensor_details->is_parameter,
               "%" << value->debugName() << " is not a Parameter");
  auto it = _mapper.ids_name_map.find(record->ipu_tensor_id);
  if (it == _mapper.ids_name_map.end()) {
    return "";
  }
  return it->second;
}

void IDispatch::replaceValue(torch::jit::Value *v_old,
                             torch::jit::Value *v_new) {
  _mapper.replaceValue(v_old, v_new);
}

// adapted from torch/csrc/jit/python/python_tracer.cpp because the header file
// had too many dependencies
torch::jit::SourceRange IDispatch::getPythonInterpreterSourceRange(
    const std::vector<torch::jit::StackEntry> &cs) const {

  auto excludes = getSourceLocationExcludes();
  const auto is_filename_excluded = [&](std::string_view filename) {
    const auto excludes_filename = [&filename](std::vector<char> exclude) {
      return filename.find(std::string_view(exclude.data(), exclude.size())) !=
             std::string_view::npos;
    };
    return std::any_of(excludes.begin(), excludes.end(), excludes_filename);
  };

  // transform_reduce
  auto stack_trace = std::accumulate(
      cs.begin(), cs.end(), std::string(),
      [](std::string trace, const torch::jit::StackEntry &entry) {
        auto file_line_col = entry.range.file_line_col();
        if (file_line_col) {
          const auto &[file, line, col] = *file_line_col;
          UNUSED(col);
          trace +=
              file + "(" + std::to_string(line) + "): " + entry.filename + "\n";
        }
        return trace;
      });

  auto val = std::find_if(
      cs.begin(), cs.end(),
      [is_filename_excluded](const torch::jit::StackEntry &entry) {
        auto file_line_col = entry.range.file_line_col();
        if (file_line_col) {
          return !is_filename_excluded(std::get<0>(*file_line_col));
        }
        return false;
      });

  c10::optional<std::string> source_filename;
  std::size_t source_line = 0;
  if (val != cs.end()) {
    std::size_t col = 0;
    std::tie(source_filename, source_line, col) = *val->range.file_line_col();
  }

  auto source = std::make_shared<torch::jit::Source>(
      stack_trace, source_filename, source_line);
  logging::trace("Setting op source to: {}:{}",
                 source_filename.value_or("<unknown>"), source_line);
  return torch::jit::SourceRange(source, 0, stack_trace.size());
}

IDispatch::~IDispatch() { resetCurrentSourceLocation(); }

} // namespace poptorch
