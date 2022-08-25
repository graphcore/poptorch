// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fstream>
#include <functional>
#include <regex>

#include "popart_compiler/CodeletsCompilation.hpp"
#include "popart_compiler/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace popart_compiler {

namespace {

// Inter-process exclusive read file lock.
class ExclusiveFileLock {
public:
  explicit ExclusiveFileLock(const std::string &path)
      : _fd(open(path.c_str(), O_RDONLY)) {
    ERROR_ON_MSG(_fd == -1, "Could not open file " << path);

    if (flock(_fd, LOCK_EX) == -1) {
      close(_fd);
      ERROR("Could not obtain an exclusive lock on file " << path);
    }
  }

  ~ExclusiveFileLock() {
    flock(_fd, LOCK_UN);
    close(_fd);
  }

private:
  int _fd;
};

// Returns the commit hash of poplar (via popc --version).
std::string poplarVersion() {
  FILE *stream = popen("popc --version", "r");
  ERROR_ON_MSG(stream == NULL,
               "Unable to read Poplar version. Is Poplar SDK enabled?");

  std::string output;
  try {
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), stream) != NULL) {
      output += buffer;
    }
  } catch (const std::exception &e) {
    pclose(stream);
    ERROR(
        "Unable to read the output of 'popc --version'. Reason: " << e.what());
  }

  ERROR_ON_MSG(pclose(stream) == -1,
               "Unable to read the output of 'popc --version'. Reason: "
                   << strerror(errno));

  std::smatch match;
  std::regex regex("([a-z0-9]{10,32})");
  if (std::regex_search(output, match, regex)) {
    return match.str();
  }

  ERROR("Unable to parse the output of 'popc --version'.");
}

// Computes a hash of the contents of a file at the specified path.
std::size_t getFileContentHash(const std::string &path) {
  std::ifstream file;
  file.open(path);
  ERROR_ON_MSG(!file.is_open(), "Could not open file " << path);

  try {
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    std::string buffer(size, '\0');
    file.seekg(0);
    file.read(&buffer[0], size);
    return std::hash<std::string>()(buffer);
  } catch (const std::exception &e) {
    ERROR("Could not read file " << path << ". Reason: " << e.what());
  }
}

// Final path is of form:
// <src_file_path_without_extension>-<src_hash>-<poplar_hash>.gp
std::string compiledCodeletPath(const std::string &src_file_path) {
  std::size_t src_hash = getFileContentHash(src_file_path);
  std::string poplar_version = poplarVersion();
  // Remove the '.inc.cpp' file extension.
  std::string out_file_path = src_file_path.substr(0, src_file_path.size() - 8);
  out_file_path += "-";
  out_file_path += std::to_string(src_hash);
  out_file_path += "-";
  out_file_path += poplar_version;
  out_file_path += ".gp";
  return out_file_path;
}

void compileCodelet(const std::string &src_file_path,
                    const std::string &out_file_path,
                    const std::string &target) {
  int pipe_fd[2];
  ERROR_ON_MSG(pipe(pipe_fd) == -1,
               "Could not compile codelet "
                   << src_file_path
                   << ", pipe failed. Reason: " << strerror(errno));

  pid_t child_pid = fork();
  ERROR_ON_MSG(child_pid == -1,
               "Could not compile codelet "
                   << src_file_path
                   << ", fork failed. Reason: " << strerror(errno));

  if (child_pid == 0) {
    // No reason to ERROR_ON_MSG as we can't see stdout/stderr at this point.
    ERROR_ON(close(pipe_fd[0]) == -1);
    ERROR_ON(setpgid(0, 0) == -1);
    // Pipe stdout and stderr to the parent process.
    ERROR_ON(dup2(pipe_fd[1], STDOUT_FILENO) == -1);
    ERROR_ON(dup2(pipe_fd[1], STDERR_FILENO) == -1);
    ERROR_ON_MSG(close(pipe_fd[1]) == -1,
                 "Could not compile codelet "
                     << src_file_path
                     << ", closing child write pipe failed. Reason: "
                     << strerror(errno));

    char *const argv[] = {const_cast<char *>("popc"),
                          const_cast<char *>("-target"),
                          const_cast<char *>(target.c_str()),
                          const_cast<char *>("-O3"),
                          const_cast<char *>(src_file_path.c_str()),
                          const_cast<char *>("-o"),
                          const_cast<char *>(out_file_path.c_str()),
                          NULL};

    std::string path_env_var = "PATH=" + std::string(std::getenv("PATH"));
    char *const env[] = {const_cast<char *>(path_env_var.c_str()), NULL};

    execvpe("popc", argv, env);
    // 'exec' only returns on failure.
    _exit(EXIT_FAILURE);
  } else {
    // Close the write end.
    ERROR_ON_MSG(close(pipe_fd[1]) == -1,
                 "Could not compile codelet "
                     << src_file_path
                     << ", closing parent write pipe failed. Reason: "
                     << strerror(errno));
    int status;
    ERROR_ON_MSG(waitpid(child_pid, &status, 0) == -1,
                 "Could not compile codelet "
                     << src_file_path
                     << ", waiting for child process failed. Reason: "
                     << strerror(errno));

    // Return on success and report errors on failures.
    std::string exit_reason;
    if (WIFEXITED(status)) {
      if (WEXITSTATUS(status) == 0) {
        // Child exited successfully.
        ERROR_ON_MSG(close(pipe_fd[0]) == -1,
                     "Could not compile codelet "
                         << src_file_path
                         << ", closing parent read pipe failed. Reason: "
                         << strerror(errno));
        return;
      }
      // Child exited with non-zero code.
      exit_reason = "child failed with exit code ";
      exit_reason += std::to_string(WEXITSTATUS(status));
      exit_reason += ".";

      // Read child stdout and stderr and let the user know what happened.
      FILE *stream = fdopen(pipe_fd[0], "r");
      std::string output;
      try {
        char buffer[1024];
        while (fgets(buffer, sizeof(buffer), stream) != NULL) {
          output += buffer;
        }
        exit_reason += " 'popc' output was:\n";
        exit_reason += output;
      } catch (const std::exception &) {
        // Only report that 'popc' command failed and ignore this error.
      }
      fclose(stream);
    } else if (WIFSIGNALED(status)) {
      // Child killed by a signal.
      exit_reason = "child killed with a signal ";
      exit_reason += std::to_string(WTERMSIG(status));
      exit_reason += " (";
      exit_reason += strsignal(WTERMSIG(status));
      exit_reason += ").";
    } else {
      exit_reason = "child failure unknown.";
    }

    close(pipe_fd[0]);
    ERROR("Could not compile codelet " << src_file_path << ", " << exit_reason);
  }
}

// True filesystem python package path where codelet sources are stored.
// It gets initialized on first 'import poptorch' from python.
std::string custom_codelets_path;

} // namespace

void setCustomCodeletsPath(const char *cache_path) {
  if (custom_codelets_path.empty()) {
    custom_codelets_path = cache_path;
  }
}

std::unique_ptr<char[]> compileCustomCodeletIfNeeded(const char *src_file_name,
                                                     bool hw_only_codelet) {
  logging::LogContext ctx("CompileCustomCodeletIfNeeded");
  logging::debug("Inspecting whether custom codelet {} needs to be compiled",
                 src_file_name);

  // Should never happen.
  ERROR_ON(custom_codelets_path.empty());

  std::string src_file_path = custom_codelets_path;
  src_file_path += "/";
  src_file_path += src_file_name;

  // Lock the src file to make sure only a single process does the compilation.
  ExclusiveFileLock lock(src_file_path);

  std::string out_file_path = compiledCodeletPath(src_file_path);

  // Skip compilation if codelet is already compiled.
  std::ifstream out_file;
  out_file.open(out_file_path);
  if (out_file.is_open()) {
    logging::debug("Custom codelet {} already compiled", src_file_name);
    out_file.close();
    return stringToUniquePtr(out_file_path);
  }

  std::string target;
  std::int64_t hw_version = ipuHardwareVersion();
  if (hw_only_codelet) {
    ERROR_ON_MSG(
        hw_version == 0 || hw_version == -1,
        "Can't infer IPU hardware version, are there any IPUs in the system?");
    target = "ipu" + std::to_string(hw_version);
  } else if (hw_version == 0 || hw_version == -1) {
    target = "cpu," + getIpuModelVersion();
  } else {
    target = "cpu,ipu" + std::to_string(hw_version);
  }

  logging::debug("Compiling custom codelet {} for target {}", src_file_name,
                 target);

  compileCodelet(src_file_path, out_file_path, target);
  return stringToUniquePtr(out_file_path);
}

} // namespace popart_compiler
} // namespace poptorch
