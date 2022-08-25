// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_CODELETS_COMPILATION_HPP
#define POPART_COMPILER_CODELETS_COMPILATION_HPP

#include <memory>

namespace poptorch {
namespace popart_compiler {

// Called from python on each 'import poptorch'. Cache path is expected to be
// a true filesystem path of the installed python package where codelet sources
// are stored.
void setCustomCodeletsPath(const char *cache_path);

// Compile a custom codelet (if not already compiled) and store the output
// file to the path specified with 'setCustomCodeletsPath' above. This can
// safely be called from multiple threads/processes.
std::unique_ptr<char[]> compileCustomCodeletIfNeeded(const char *src_file_name,
                                                     bool hw_only_codelet);

} // namespace popart_compiler
} // namespace poptorch

#endif // POPART_COMPILER_CODELETS_COMPILATION_HPP
