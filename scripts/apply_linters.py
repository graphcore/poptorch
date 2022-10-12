#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import collections
import difflib
import enum
import json
import logging
import os
import pathlib
import re
import sys
import tempfile
import time
import packaging.version
import yaml

from utils import _utils

logger = logging.getLogger("apply_linters")
_utils.set_logger(logger)

yapf_flags = "--style='{based_on_style: pep8}'"
cpp_lint_disabled = [
    "runtime/string", "runtime/references", "build/c++11",
    "build/header_guard", "whitespace/comments", "whitespace/indent"
]


class GitStrategy(enum.Enum):
    Master = "master"  # Files modified / added between HEAD and origin/master
    Head = "head"  # Files modified / added in the last commit
    Diff = "diff"  # Files modified / added but not commited
    All = "all"  # All files tracked by git


class ILinterFamily:
    """Regroup the linters running on the same types of files (e.g cpp or py)
    """

    def __init__(self, supported_extensions, linters,
                 excluded_extensions=None):
        """
        :param supported_extensions: Array of extensions supported by the
            linters (e.g ["hpp","cpp"])
        :param linters: list of linters to run for the matching files
        :param excluded_extensions: Optional list of extensions to exclude
        """
        self._linters = linters
        # ["hpp","cpp"] -> ".*\.(hpp|cpp)$"
        self._supported = re.compile(r".*\.(%s)$" %
                                     '|'.join(supported_extensions))
        if excluded_extensions:
            self._excluded = re.compile(r".*\.(%s)$" %
                                        '|'.join(excluded_extensions))
        else:
            self._excluded = None
        self.first_lint = True

    def gen_lint_commands(self, filename, autofix):
        if not re.match(self._supported, filename):
            logger.debug("%s didn't match %s", filename, self._supported)
            return []
        if self._excluded and re.match(self._excluded, filename):
            logger.debug("%s matched exclusion %s", filename, self._excluded)
            return []
        if self.first_lint:
            # Check all the linters are correctly installed
            self.first_lint = False
            all_valid = all(
                [linter.check_version() for linter in self._linters])
            if not all_valid:
                print("\nERROR: You need a valid PopTorch buildenv to run "
                      "the linters:")
                print("- create a buildenv using scripts/create_buildenv.py")
                print(
                    "- activate the environment: source activate_buildenv.sh")
                print("- configure your PopTorch build: cmake "
                      "../poptorch -DPOPLAR_SDK=...")
                sys.exit(1)

        return [
            linter.gen_lint_command(filename, autofix)
            for linter in self._linters
            if linter.is_enabled(filename, autofix)
        ]


class CppLinters(ILinterFamily):
    def __init__(self):
        super().__init__(["hpp", "cpp"],
                         excluded_extensions=["inc.hpp", "inc.cpp"],
                         linters=[ClangTidy(),
                                  ClangFormat(),
                                  CppLint()])


class PyLinters(ILinterFamily):
    def __init__(self):
        super().__init__(["py"], linters=[Pylint(), Yapf()])

    def is_enabled(self, filename, autofix):  # pylint: disable=unused-argument
        # Don't run PyLint on the buildenv config files
        return re.match(r".*\.buildenv\.py$", filename) is None


class ILinter:
    """Base class for all the linters"""

    def gen_lint_command(self, filename, autofix):
        """Create one or more commands to lint the given file"""
        raise RuntimeError("Must be implemented by child class")

    def check_version(self):
        """Check the linter is installed. (Called only once)"""
        raise RuntimeError("Must be implemented by child class")

    def is_enabled(self, filename, autofix):  # pylint: disable=unused-argument
        """Should the linter run for this given file?"""
        return True


class ProcessManager:
    _manager = None

    @staticmethod
    def create(max_num_proc=0):
        assert ProcessManager._manager is None
        ProcessManager._manager = ProcessManager(max_num_proc)

    @staticmethod
    def get():
        if ProcessManager._manager is None:
            ProcessManager.create()
        return ProcessManager._manager

    def __init__(self, max_num_proc):
        self.max_num_proc = max_num_proc
        self.queue = []
        self.running = []
        self.num_running = 0

    def enqueue(self, create_proc_fn):
        if self.max_num_proc == 0:
            create_proc_fn()
            return

        self.queue.append(create_proc_fn)
        self.update()

    def update(self):
        def _is_running(proc):
            """Update num_running when a process just returned
            """
            if proc.is_running():
                return True
            self.num_running -= 1
            logger.debug("Process completed, %d/%d processes in use",
                         self.num_running, self.max_num_proc)
            return False

        # Check the status of all the running processes
        self.running = [p for p in self.running if _is_running(p)]

        # Start new processes if slots are available
        while self.queue and self.num_running < self.max_num_proc:
            self.running.append(self.queue[0]())
            self.queue = self.queue[1:]
            self.num_running += 1
            logger.debug("Process started, %d/%d processes in use",
                         self.num_running, self.max_num_proc)


class Command:
    """Asynchronously run a command in a sub shell"""

    def __init__(self,
                 *cmd,
                 stop_on_error=True,
                 print_output=True,
                 output_processor=None,
                 name=None,
                 print_output_on_error=True):
        # Stop on error
        self.cmd = "set -e;" if stop_on_error else ""
        self.cmd += " ".join(cmd)
        self.output_processor = output_processor
        self.print_output = print_output
        self.proc = None
        self.output = ""
        self.name = name or cmd[0]
        self.print_output_on_error = print_output_on_error

    def start(self):
        ProcessManager.get().enqueue(self._create_proc)

    def _create_proc(self):
        assert self.proc is None, "Process already started"
        self.output = ""

        def append_to_output(line):
            self.output += line + "\n"

        # We make sure that the PYTHONPATH is clear because we do not want the
        # linter to undertake run-time inspection of the poptorch module.
        new_env = os.environ.copy()
        new_env["PYTHONPATH"] = ""
        if "CPATH" in new_env:
            del new_env["CPATH"]

        self.proc = _utils.Process([self.cmd],
                                   redirect_stderr=True,
                                   env=new_env,
                                   stdout_handler=append_to_output)
        return self.proc

    def is_running(self):
        return self.proc is None or self.proc.is_running()

    def wait(self):
        while self.proc is None:
            ProcessManager.get().update()
            time.sleep(1)
        returncode = self.proc.wait()
        output = self.output

        logger.debug("Command %s returned with %d", self.name, returncode)
        if self.output_processor:
            output, returncode = self.output_processor(output, returncode)
        if self.print_output_on_error and returncode:
            print(f"{self.name} failed with exit code {returncode}")
            print("Output:")
            print(output)
        elif self.print_output and output:
            print(f"Output of {self.name}:")
            print(output)
        return returncode

    def run(self):
        self.start()
        return self.wait()


class CondaCommand(Command):
    """A command which will activate a Conda buildenv before running"""
    activate_cmd = None

    def __init__(self, *cmd, name=None, **kwargs):
        if CondaCommand.activate_cmd is None:
            CondaCommand.activate_cmd = get_conda_activate_cmd()
            logger.debug("Activate command initialised to %s",
                         CondaCommand.activate_cmd)
        if cmd:
            super().__init__(CondaCommand.activate_cmd,
                             *cmd,
                             **kwargs,
                             name=name or cmd[0])


def get_llvm_path_from_build():
    llvm_path = ""

    def parse_cmake_cache(output, returncode):
        # Expected to contain a line of the form:
        # LLVM_DIR:PATH=/path/to/llvm_dir/lib/cmake/llvm
        nonlocal llvm_path
        for line in output.splitlines():
            m = re.match("^LLVM_DIR:.*=(.*)/lib/cmake/llvm", line)
            if m:
                llvm_path = m.group(1)
                break
        return output, returncode

    CondaCommand("cat ${CONDA_PREFIX}/../CMakeCache.txt",
                 print_output=False,
                 output_processor=parse_cmake_cache).run()
    return llvm_path


class ClangTools:
    _llvm_path = None

    @staticmethod
    def path():
        if ClangTools._llvm_path is None:
            ClangTools._llvm_path = get_llvm_path_from_build()
        return os.path.join(ClangTools._llvm_path, "bin")

    @staticmethod
    def clang_format():
        return os.path.join(ClangTools.path(), "clang-format")

    @staticmethod
    def clang_tidy():
        return os.path.join(ClangTools.path(), "clang-tidy")

    @staticmethod
    def clang_apply_replacements():
        return os.path.join(ClangTools.path(), "clang-apply-replacements")


def get_conda_activate_cmd():
    """Check if we're already inside a Conda environment, if not return the
    command to run to activate one"""
    if "CONDA_PREFIX" in os.environ:
        logger.debug("Conda environment active, nothing to do")
        return ""
    sources_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    activate_script = os.path.join(sources_dir, ".linters",
                                   "activate_buildenv.sh")
    if not os.path.isfile(activate_script):
        error = ["No active Conda environment, you need to either activate "\
                    "it or create a link to it:",
                 ". ../build/activate_buildenv.sh",
                 "or",
                 f"ln -sf /my/build/activate_buildenv.sh {activate_script}"
                ]
        raise RuntimeError("\n".join(error))
    return f". {activate_script};"


def offset_to_line(filename, offsets):
    """Convert a list of offsets in a file to a dictionary of line, column.
    [ offset ] -> { offset: (line,column) }
    """
    if not filename:
        return {offset: (0, 0) for offset in offsets}
    offsets = sorted(set(offsets))
    line = 1
    mappings = {}
    file_offset = 0
    try:
        it = iter(offsets)
        offset = next(it)
        for l in open(filename):
            start_line_offset = file_offset
            file_offset += len(l)
            while offset < file_offset:
                mappings[offset] = (line, offset - start_line_offset + 1)
                offset = next(it)
            line += 1
    except StopIteration:
        return mappings
    raise RuntimeError(f"Invalid offset {offset} (File length: {file_offset})")


class DiffCreator:
    """Create a diff between the output of a command and the content of file.
    Some linters (for example yapf) print the modified file to stdout instead
    of modifying it in-place.
    This class will create a diff with the original file and print the
    differences.
    If autofix is enabled, the content of the original file
    will be replaced.
    """

    def __init__(self, filename, linter, autofix):
        self.filename = filename
        self.linter = linter
        if autofix:
            self.linter += "(autofix)"
        self.autofix = autofix

    def __call__(self, output, errcode):
        """Called by Command with the output of the linter"""
        origin = open(self.filename).readlines()
        new = output.splitlines(True)
        delta = ""
        for line in difflib.unified_diff(origin,
                                         new,
                                         fromfile="a/" + self.filename,
                                         tofile="b/" + self.filename):
            m = re.match(r"@@ -(\d+),.*@@", line)
            if m:
                print(f"{self.filename}:{int(m.group(1))+3}:error:"
                      f"[{self.linter}] to fix run "
                      "./scripts/apply_linters.py --autofix")
            delta += line
        if delta:
            if self.autofix:
                with open(self.filename, "w") as f:
                    f.write(output)
            else:
                print(f"{self.linter} found the following issues in "
                      f"{self.filename}\n{delta}")
            errcode = 1
        return delta, errcode


class VersionParseCommandBase(CondaCommand):
    def __init__(self, *cmd, **kwargs):
        super().__init__(*cmd, **kwargs)
        self.version = None

    def _parse_version(self, output, return_code):
        raise NotImplementedError("Must be implemented in the derived type")

    def run_and_compare_versions(self, expected):
        self.run()
        expected_version = packaging.version.parse(expected)

        if expected_version != self.version:
            logger.error("Required version of %s is %s, but found %s",
                         self.name, expected_version, self.version)
            return False
        return True


class VersionJSONParseCommand(VersionParseCommandBase):
    def __init__(self, command_name):
        super().__init__(
            f"grep \\\"version\\\" "
            f"${{CONDA_PREFIX}}/conda-meta/{command_name}-*.json",
            print_output=False,
            output_processor=self._parse_version)

    def _parse_version(self, output, return_code):
        if return_code:
            return output, return_code
        self.version = packaging.version.parse(
            json.loads("{" + output + "}")["version"])
        return output, return_code


class VersionParseCommand(VersionParseCommandBase):
    def __init__(self, version_re_prefix, command_name):
        super().__init__(command_name,
                         "--version",
                         print_output=False,
                         output_processor=self._parse_version)
        self.version_re_prefix = version_re_prefix
        self.version = None

    def _parse_version(self, output, return_code):
        if return_code:
            return output, return_code

        match_result = re.search(f"{self.version_re_prefix} ([.0-9]+)",
                                 output,
                                 flags=re.MULTILINE)

        if match_result:
            self.version = packaging.version.parse(match_result[1])

        return output, return_code


def compare_versions_from_conda(command_name, expected):
    version_parse_cmd = VersionJSONParseCommand(command_name)
    return version_parse_cmd.run_and_compare_versions(expected)


def compare_versions_from_output(command_name,
                                 expected,
                                 version_re_prefix=None):
    if version_re_prefix is None:
        version_re_prefix = command_name

    version_parse_cmd = VersionParseCommand(version_re_prefix, command_name)
    return version_parse_cmd.run_and_compare_versions(expected)


class ClangFormat(ILinter):
    def gen_lint_command(self, filename, autofix):
        flags = ""
        output_processor = None
        if autofix:
            flags += " -i"
        else:
            output_processor = DiffCreator(filename, "clang-format", autofix)

        return Command(ClangTools.clang_format(),
                       flags,
                       filename,
                       output_processor=output_processor,
                       print_output=autofix)

    def check_version(self):
        return compare_versions_from_output(ClangTools.clang_format(),
                                            "15.0.2", "version")


class ClangTidy(ILinter):
    class ResultsProcessor:
        """Wait for all the jobs to complete then combine and process their
        outputs
        """

        def __init__(self, num_jobs, autofix):
            self.num_jobs = num_jobs
            self.tmp_folder = tempfile.TemporaryDirectory(
                prefix="poptorchLinter_")
            self.autofix = autofix

        def __call__(self, raw_output, returncode):
            self.num_jobs -= 1
            logger.debug("1 clang-tidy job completed, %d remaining",
                         self.num_jobs)
            logger.debug("clang-tidy output: %s", raw_output)
            if self.num_jobs == 0:
                diagnostics = []
                # Combine the diagnostics from the different reports
                for f in pathlib.Path(self.tmp_folder.name).glob("*.yaml"):
                    with open(f) as file:
                        res = yaml.full_load(file)
                        # Combine the diagnostics
                        diagnostics += res.get("Diagnostics", [])

                # Error messages are linked to a file + offset
                # Collect the "offsets" used for each filename
                offsets = collections.defaultdict(list)
                for diag in diagnostics:
                    msg = diag["DiagnosticMessage"]
                    offsets[msg["FilePath"]].append(msg["FileOffset"])

                # Create a map of map that linking offsets in files to their
                # corresponding line and column:
                # line_mapping[filename] = { offset : (line, col) }
                line_mappings = {}
                for filename, file_offsets in offsets.items():
                    # Don't lint files in the build folder
                    if not os.path.isabs(filename):
                        continue
                    line_mappings[filename] = offset_to_line(
                        filename, file_offsets)
                printed = []
                for diag in diagnostics:
                    msg = diag["DiagnosticMessage"]
                    filename = msg["FilePath"]
                    # Don't lint files in the build folder
                    if not os.path.isabs(filename):
                        continue
                    line, col = line_mappings[filename][msg["FileOffset"]]
                    error = "error"
                    if self.autofix and msg["Replacements"]:
                        error += " (autofixed)"
                    output = f"{filename}:{line}:{col}: {error}: "
                    output += f"{msg['Message']} [{diag['DiagnosticName']}]"

                    # If this message has already been printed: skip it
                    if output in printed:
                        continue
                    if not printed:
                        print("Output of clang-tidy:")
                    print(output)
                    printed.append(output)
                if not printed and returncode != 0:
                    # If we didn't manage to parse the diagnostics but clang-tidy
                    # returned a failure at least print the raw output.
                    print(raw_output)
                # Apply the fixes using clang-apply-replacements
                if self.autofix:
                    Command(ClangTools.clang_apply_replacements(),
                            self.tmp_folder.name).run()
            return raw_output, returncode

    def __init__(self):
        self.configs = []
        self.includes = []
        self.compile_commands = {}

    def get_compile_commands_flags(self, filename):
        if filename.endswith("cpp"):
            if filename in self.compile_commands:
                return self.compile_commands[filename]
            logger.warning(
                "%s is absent from compile_commands.json: check "
                "CMakeLists.txt to make sure it's compiled", filename)
            # Fall through to header path to try to find
            # flags for files in the same folder

        folder = os.path.dirname(filename)
        filename = os.path.basename(filename)
        path = folder.split(os.path.sep)
        # If it's a public header then it will be in
        # poptorch/component/include/component/my_header.hpp
        # and the cpp files will be in /component/source/
        #
        # Therefore we need to replace "include/component" with "source"
        # to find a cpp file with the compilation flags we want.
        if "include" in path:
            # Remove folders in path up to "include"
            while path.pop() != "include":
                continue

            # Types is a sub module in popart_compiler, so we want to go up one more level.
            if path[-1] == "types":
                path.pop()

            # TODO(T49191) lower_to_poplar, dialect and pytorch_bridge don't
            # have their sources in a "source" subfolder at the moment.
            exceptions = ["lower_to_poplar", "pytorch_bridge", "dialect"]
            if not "source" in path and not any(comp in path
                                                for comp in exceptions):
                # Point at "source" instead
                path.append("source")
        # else it's a private header: nothing to do, it's already in the same
        # folder as the source files.
        folder = os.path.join(*path)

        for path, flags in self.compile_commands.items():
            if path.startswith(folder):
                logger.debug("Found flags for folder %s", folder)
                return flags
        logger.warning("No compilation flags found for folder %s", folder)
        return ("", "")

    def gen_lint_command(self, filename, autofix):
        if not self.configs:
            self.check_version()
        gcc_flags, work_dir = self.get_compile_commands_flags(filename)
        flags = "-std=c++17 -fsized-deallocation -DONNX_NAMESPACE=onnx "
        flags += gcc_flags
        flags += " -I" + " -I".join(self.includes)
        cd = ""
        if work_dir:
            cd = f"cd {work_dir};"

        commands = []
        results = ClangTidy.ResultsProcessor(len(self.configs), autofix)
        # Clang-tidy has a lot of checks so we run them in parallel in
        # different processes
        for i, c in enumerate(self.configs):
            report = os.path.join(results.tmp_folder.name, f"report_{i}.yaml")
            commands.append(
                Command(cd,
                        ClangTools.clang_tidy(),
                        "--quiet",
                        os.path.realpath(filename),
                        f"--export-fixes={report}",
                        c,
                        "--",
                        flags,
                        name=("clang-tidy --quiet "
                              f"{filename} -- {flags}"),
                        output_processor=results,
                        print_output_on_error=False,
                        print_output=False))
        return commands

    def process_compile_commands(self, commands):
        # Some flags are not supported by clang-tidy
        unsupported_flags = ["-fno-semantic-interposition"]
        for c in commands:
            gcc_flags = c["command"].split()
            cmd = " ".join(
                [f for f in gcc_flags if f not in unsupported_flags])
            m = re.match(".*/poptorch/(.*)", c["file"])
            assert m, f"Couldn't find '/poptorch/' in {c['file']}"

            # Exception we've got nested "poptorch" folders, so make sure
            # the path is the correct one.
            file_maybe = m.group(1)
            if not os.path.exists(file_maybe):
                file_maybe = os.path.join("poptorch", file_maybe)

            if not os.path.exists(file_maybe):
                logger.warning(
                    "compile_commands.json: %s/%s ignored: neither file exist",
                    m.group(1), file_maybe)
            self.compile_commands[file_maybe] = (cmd, c["directory"])

    # pylint: disable=too-many-return-statements
    def check_version(self):
        config = []
        self.configs = []

        def parse_config(output, returncode):
            nonlocal config
            config = output.splitlines(True)
            # For some reason clang-tidy's config contains these options it doesn't support, so filter them out.
            excludes = [
                "FunctionHungarianPrefix", "MethodHungarianPrefix",
                "NamespaceHungarianPrefix"
            ]
            config = [
                line for line in config
                if not any([e in line for e in excludes])
            ]
            return output, returncode

        def parse_checks(output, returncode):
            nonlocal config
            # Ignore first line it's the header
            all_checks = output.splitlines()[1:]
            checks_per_thread = 40
            for offset in range(0, len(all_checks), checks_per_thread):
                checks = all_checks[offset:offset + checks_per_thread]
                config[1] = "Checks: '" + ",".join(checks) + "'\n"
                self.configs.append("--config=\"" + "".join(config) + "\"")
            return output, returncode

        def parse_include_tests(output, returncode):
            if output:
                returncode = 1
            return output, returncode

        def parse_system_includes(output, returncode):
            if returncode:
                logger.error("Failed to find system includes: %s", output)
                return output, returncode
            include_path_section = False
            for line in output.split("\n"):
                if "search starts here" in line:
                    include_path_section = True
                if include_path_section and line.startswith(" "):
                    logger.debug("Adding %s to includes", line)
                    self.includes.append(line.rstrip())
            return output, returncode

        def parse_compile_commands_file(output, returncode):
            if returncode:
                logger.error("compile_commands.json not found. "
                             "Make sure to build PopTorch first.")
                return output, returncode

            self.process_compile_commands(json.loads(output))
            return output, returncode

        if CondaCommand("g++ -E -x c++ - -v < /dev/null",
                        print_output=False,
                        output_processor=parse_system_includes).run():
            return False

        if Command(ClangTools.clang_tidy() + " --dump-config",
                   print_output=False,
                   output_processor=parse_config).run():
            return False
        if Command(ClangTools.clang_tidy() + " --list-checks",
                   print_output=False,
                   output_processor=parse_checks).run():
            return False
        tests = [
            f"test -d {i} || echo \"Include folder {i} not found\""
            for i in self.includes
        ]
        if CondaCommand(";".join(tests),
                        stop_on_error=False,
                        output_processor=parse_include_tests).run():
            return False

        # Check if there is a compile_commands.json
        if CondaCommand("cat ${CONDA_PREFIX}/../compile_commands.json",
                        print_output=False,
                        output_processor=parse_compile_commands_file).run():
            return False

        return compare_versions_from_output(ClangTools.clang_tidy(), "15.0.2",
                                            "version")

    def is_enabled(self, filename, autofix):
        # Don't run Clang Tidy on the pybind11 modules because we don't know
        # where pybind headers are.
        return "custom_cube_op.cpp" not in filename and \
                "python/" not in filename


class CppLint(ILinter):
    def cpplint(self):
        return "${CONDA_PREFIX}/bin/cpplint"

    def gen_lint_command(self, filename, autofix):
        return CondaCommand(self.cpplint(), "--root=include --quiet",
                            f"--filter=-{',-'.join(cpp_lint_disabled)}",
                            filename)

    def check_version(self):
        return compare_versions_from_output(self.cpplint(), "1.4.4", "cpplint")


class Pylint(ILinter):
    def pylint(self):
        return "${CONDA_PREFIX}/bin/pylint"

    def gen_lint_command(self, filename, autofix):
        return CondaCommand(
            self.pylint(), "--score=no --reports=no -j 0 --msg-template="
            "'{path}:{line}:{column}:error:pylint[{symbol}({msg_id})]: {msg}'"
            " --rcfile=.pylintrc", filename)

    def check_version(self):
        return compare_versions_from_output(self.pylint(), "2.5.3", "pylint")

    def is_enabled(self, filename, autofix):  # pylint: disable=unused-argument
        # Don't run PyLint on the buildenv config files
        return re.match(r".*\.buildenv\.py$", filename) is None


class Yapf(ILinter):
    def yapf(self):
        return "${CONDA_PREFIX}/bin/yapf"

    def gen_lint_command(self, filename, autofix):
        flags = yapf_flags
        output_processor = None
        if autofix:
            flags += " -i"
        else:
            output_processor = DiffCreator(filename, "yapf", autofix)

        return CondaCommand(self.yapf(),
                            flags,
                            filename,
                            output_processor=output_processor,
                            print_output=autofix)

    def check_version(self):
        return compare_versions_from_output(self.yapf(), "0.27.0", "yapf")


class Executor:
    def __init__(self, filename, cmd):
        self.filename = filename
        self.cmd = cmd
        self.returncode = 0
        self._next_step()

    def _next_step(self):
        for step in self.cmd[0]:
            step.start()

    def update(self):
        for s in self.cmd[0]:
            if s.is_running():
                return
        # All steps complete for this command:
        for s in self.cmd[0]:
            self.returncode += s.wait()
        self.cmd = self.cmd[1:]
        if self.cmd:
            self._next_step()
        elif self.returncode:
            print(f"{self.filename}:error: contains linting errors: "
                  "run ./scripts/apply_linters.py --autofix")

    def execution_complete(self):
        return not self.cmd


class Linters:
    """Interface class used to lint files"""

    def __init__(self):
        self._linters = [CppLinters(), PyLinters()]

    def lint_git(self, strategy, autofix):
        files = []

        class GetFiles:
            def __init__(self, files):
                self.files = files

            def __call__(self, output, returncode):
                # If we keep the last element of each line we will have the files we need to lint.
                # ['M', 'poptorch/source/dispatch_tracer/RegisterAtenOverloads.cpp']
                # ['R092', 'poptorch/source/dispatch_tracer/dispatchers/Tracer.hpp', 'poptorch/source/dispatch_tracer/dispatchers/IDispatch.hpp']
                # ['A', 'poptorch/source/dispatch_tracer/dispatchers/JitDispatch.hpp']
                for line in output.splitlines():
                    self.files.append(line.split()[-1])
                return output, returncode

        assert isinstance(strategy, GitStrategy)
        git_cmd = ""
        filter_cmd = "| grep \"^[AMRT]\" "
        if strategy == GitStrategy.Master:
            git_cmd = "git diff --name-status -r origin/master "
        elif strategy == GitStrategy.Head:
            git_cmd = "git diff --name-status -r HEAD^ "
        elif strategy == GitStrategy.Diff:
            git_cmd = "git diff --name-status -r HEAD "
        elif strategy == GitStrategy.All:
            git_cmd = "git ls-tree --name-only -r HEAD "
            filter_cmd = ""
        else:
            raise RuntimeError("Unknown strategy requested")
        Command(git_cmd,
                filter_cmd,
                print_output=False,
                output_processor=GetFiles(files)).run()
        self.lint_files(files, autofix)

    def lint_files(self, files, autofix):
        jobs = {}
        for f in files:
            cmd = self._gen_lint_commands(f, autofix)
            if cmd:
                jobs[f] = cmd
        if not jobs:
            logger.info("No linter to run: early return")
            return
        executors = []
        for filename, cmd in jobs.items():
            print(f"Linting file {filename} [{len(cmd)}] commands to run")
            if autofix:
                executors.append(Executor(filename, cmd))
            else:
                # No risk of conflicting modification in place
                # Merge the steps from all the linters
                all_steps = []
                for c in cmd:
                    all_steps += c
                executors.append(Executor(filename, [all_steps]))
        still_running = True
        while still_running:
            still_running = False
            ProcessManager.get().update()
            for e in executors:
                if e.execution_complete():
                    continue
                e.update()
                still_running = True
            time.sleep(1)

    def _gen_lint_commands(self, filename, autofix):
        cmd = []
        for linter in self._linters:
            cmd += linter.gen_lint_commands(filename, autofix)
        return [[c] if isinstance(c, Command) else c for c in cmd]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO Add option to exclude some linters (e.g -no-clang-tidy)
    # TODO Check / update Copyrights
    parser.add_argument("--debug",
                        "-d",
                        action="store_true",
                        help="Print debug messages")
    parser.add_argument("--autofix",
                        "-a",
                        action="store_true",
                        help="Automatically apply fixes when possible")
    parser.add_argument(
        "--git-strategy",
        "-s",
        type=str,
        choices=[v.value for _, v in GitStrategy.__members__.items()],
        default=GitStrategy.Master.value,
        help="Strategy to use when no files are passed")
    parser.add_argument("--jobs",
                        "-j",
                        type=int,
                        default=_utils.get_nprocs(),
                        help="Number of cores to use for linting (0 = auto)")
    parser.add_argument("files", nargs="*", help="one or more files to lint")
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    logger.debug("Args: %s", str(args))

    if args.jobs:
        assert args.jobs >= 0
        ProcessManager.create(args.jobs)

    # Check we've got a Conda environment available
    CondaCommand()

    linters = Linters()
    if args.files:
        linters.lint_files(args.files, args.autofix)
    else:
        print(
            f"Linting files selected by the git strategy '{args.git_strategy}'"
        )
        linters.lint_git(GitStrategy(args.git_strategy), args.autofix)
