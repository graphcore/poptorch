#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import collections
import difflib
import enum
import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import yaml

logger = logging.getLogger("apply_linters")

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
                print("\nERROR: You need a valid Poptorch buildenv to run "
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
        super().__init__(["hpp", "cpp", "inc"],
                         linters=[ClangTidy(),
                                  ClangFormat(),
                                  CppLint()])


class PyLinters(ILinterFamily):
    def __init__(self):
        super().__init__(["py"], linters=[Pylint(), Yapf()])


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


class Command:
    """Asynchronously run a command in a sub shell"""

    def __init__(self,
                 *cmd,
                 stop_on_error=True,
                 print_output=True,
                 output_processor=None,
                 name=None):
        # Stop on error
        self.cmd = "set -e;" if stop_on_error else ""
        self.cmd += " ".join(cmd)
        self.output_processor = output_processor
        self.print_output = print_output
        self.proc = None
        self.name = name or cmd[0]

    def start(self):
        assert self.proc is None, "Process already started"
        self.proc = subprocess.Popen([self.cmd],
                                     shell=True,
                                     executable='/bin/bash',
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)

    def is_running(self):
        assert self.proc, "Process not started"
        return self.proc.poll() is None

    def wait(self):
        assert self.proc, "Process not started"
        self.proc.wait()

        output = self.proc.stdout.read().decode("utf-8")
        returncode = self.proc.returncode
        logger.debug("Command %s returned with %d",
                     self.cmd.split("\n")[0], returncode)
        if self.output_processor:
            output, returncode = self.output_processor(output, returncode)
        if self.print_output and output:
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
                      f"formatting error[{self.linter}]")
            delta += line
        if delta and self.autofix:
            with open(self.filename, "w") as f:
                f.write(output)
            errcode = 1
        return delta, errcode


class ClangFormat(ILinter):
    def gen_lint_command(self, filename, autofix):
        flags = ""
        output_processor = None
        if autofix:
            flags += " -i"
        else:
            output_processor = DiffCreator(filename, "clang-format", autofix)

        return CondaCommand("clang-format",
                            flags,
                            filename,
                            output_processor=output_processor,
                            print_output=autofix)

    def check_version(self):
        return True  # TODO(T32437)


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

        def __call__(self, output, returncode):
            self.num_jobs -= 1
            logger.debug("1 clang-tidy job completed, %d remaining",
                         self.num_jobs)
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
                    line_mappings[filename] = offset_to_line(
                        filename, file_offsets)
                printed = []
                for diag in diagnostics:
                    msg = diag["DiagnosticMessage"]
                    filename = msg["FilePath"]
                    line, col = line_mappings[filename][msg["FileOffset"]]
                    error = "error"
                    if self.autofix and msg["Replacements"]:
                        error += " (autofixed)"
                    output = f"{filename}:{line}:{col}: {error}: "
                    output += f"{msg['Message']} [{diag['DiagnosticName']}]"

                    # If this message has already been printed: skip it
                    if output in printed:
                        continue
                    print(output)
                    printed.append(output)
                # Apply the fixes using clang-apply-replacements
                if self.autofix:
                    CondaCommand("clang-apply-replacements",
                                 self.tmp_folder.name).run()
            return output, returncode

    def __init__(self):
        self.configs = []
        self.python_includes = ""
        self.includes = [
            "${CONDA_PREFIX}/include", "${CONDA_PREFIX}/../poplar",
            "${CONDA_PREFIX}/../popart", "popart_compiler/include",
            "poptorch/include", "poptorch_logging/include"
        ]

    def gen_lint_command(self, filename, autofix):
        if not self.configs:
            self.check_version()
        flags = "-std=c++17 -fsized-deallocation -DONNX_NAMESPACE=onnx "
        flags += self.python_includes
        flags += " -I" + " -I".join(self.includes)
        commands = []
        results = ClangTidy.ResultsProcessor(len(self.configs), autofix)
        # Clang-tidy has a lot of checks so we run them in parallel in
        # different processes
        for i, c in enumerate(self.configs):
            report = os.path.join(results.tmp_folder.name, f"report_{i}.yaml")
            commands.append(
                CondaCommand("clang-tidy",
                             "--quiet",
                             filename,
                             f"--export-fixes={report}",
                             c,
                             "--",
                             flags,
                             output_processor=results,
                             print_output=False))
        return commands

    def check_version(self):
        config = []
        self.configs = []

        def parse_config(output, returncode):
            nonlocal config
            config = output.splitlines(True)
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

        def parse_python_includes(output, returncode):
            self.python_includes = output.rstrip()
            return output, returncode

        def append_to_includes(output, returncode):
            logger.debug("Adding %s to includes", output)
            self.includes.append(output)
            return output, returncode

        def parse_include_tests(output, returncode):
            if output:
                returncode = 1
            return output, returncode

        if CondaCommand("python3-config --includes",
                        print_output=False,
                        output_processor=parse_python_includes).run():
            return False
        if CondaCommand(
                "python3 -c \"import os,torch; from pathlib import Path;",
                "print(os.path.join(Path(torch.__file__).parent,\'include'),",
                " end='')\"",
                print_output=False,
                output_processor=append_to_includes).run():
            return False
        if CondaCommand("clang-tidy",
                        "--dump-config",
                        print_output=False,
                        output_processor=parse_config).run():
            return False
        if CondaCommand("clang-tidy",
                        "--list-checks",
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
        #TODO(T32437) check for clang-apply-replacements
        return True  # TODO(T32437): check version

    def is_enabled(self, filename, autofix):
        return "custom_cube_op.cpp" not in filename


class CppLint(ILinter):
    def gen_lint_command(self, filename, autofix):
        return CondaCommand("cpplint", "--root=include",
                            f"--filter=-{',-'.join(cpp_lint_disabled)}",
                            filename)

    def check_version(self):
        return True  # TODO(T32437)


class Pylint(ILinter):
    def gen_lint_command(self, filename, autofix):
        return CondaCommand(
            "pylint", "--score=no --reports=no -j 0 --msg-template="
            "'{path}:{line}:{column}:error:pylint[{symbol}({msg_id})]: {msg}'"
            " --rcfile=.pylintrc", filename)

    def check_version(self):
        return True  # TODO(T32437)


class Yapf(ILinter):
    def gen_lint_command(self, filename, autofix):
        flags = yapf_flags
        output_processor = None
        if autofix:
            flags += " -i"
        else:
            output_processor = DiffCreator(filename, "yapf", autofix)

        return CondaCommand("yapf",
                            flags,
                            filename,
                            output_processor=output_processor,
                            print_output=autofix)

    def check_version(self):
        return True  # TODO(T32437)


def arc_clang_tidy(args):
    os.system("clang-tidy " + " ".join(args))


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
            print(f"{self.filename}:error: contains linting errors")

    def execution_complete(self):
        return not self.cmd


class Linters:
    """Interface class used to lint files"""

    def __init__(self):
        self._cpp_linters = CppLinters()
        self._py_linters = PyLinters()

    def lint_git(self, strategy, autofix):
        files = []

        class GetFiles:
            def __init__(self, files):
                self.files = files

            def __call__(self, output, returncode):
                self.files += output.splitlines()
                return output, returncode

        assert isinstance(strategy, GitStrategy)
        git_cmd = ""
        filter_cmd = "| grep \"^[AMRT]\" "
        filter_cmd += "| cut -f 2"
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
            for e in executors:
                if e.execution_complete():
                    continue
                e.update()
                still_running = True
            time.sleep(1)

    def _gen_lint_commands(self, filename, autofix):
        cmd = self._cpp_linters.gen_lint_commands(
            filename, autofix) + self._py_linters.gen_lint_commands(
                filename, autofix)
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
    #TODO(T32438): Not implemented
    parser.add_argument("--jobs",
                        "-j",
                        type=int,
                        default=0,
                        help="Number of cores to use for linting (0 = auto)")
    parser.add_argument("files", nargs="*", help="one or more files to lint")
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    logger.debug("Args: %s", str(args))

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
