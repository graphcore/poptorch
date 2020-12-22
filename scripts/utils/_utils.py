# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import re
import enum
import fcntl
import json
import os
import shutil
import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


# Make the _utils functions log using the caller's logger instead of the
# default 'utils/_utils.py'
def set_logger(new_logger):
    global logger
    logger = new_logger


def rmdir_if_exists(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)


def rm_if_exists(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def get_first_line(filename):
    return open(filename, "r").readline().rstrip()


def sources_dir():
    # ./scripts/utils/../../:
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class OsType(enum.Enum):
    Osx = "osx"
    Linux = "linux"
    Unknown = "unknown"


def get_required_torch_version():
    for line in open(os.path.join(sources_dir(), "CMakeLists.txt"), "r"):
        m = re.match(r"set\(TORCH_VERSION +([0-9.]+)\)", line)
        if m:
            return m.group(1)
    raise RuntimeError("Couldn't find TORCH_VERSION in CMakeLists.txt")


class PkgInfo:
    _pkg_info_file = "pkg_info.json"

    def __init__(self,
                 version=None,
                 snapshot=None,
                 os_type=None,
                 package_os_type=None,
                 build_number=None,
                 doc_name=None,
                 project_name=None,
                 **kwargs):
        logger.debug(
            "PkgInfo: user provided version=%s snapshot=%s os_type=%s"
            " package_os_type=%s build_number=%s doc_name=%s "
            "project_name=%s", version, snapshot, os_type, package_os_type,
            build_number, doc_name, project_name)
        self.version = version or _get_version()
        self.snapshot = snapshot or _get_snapshot()
        self.os_type = os_type or get_os_type()
        if isinstance(self.os_type, OsType):
            self.os_type = self.os_type.value
        self.package_os_type = package_os_type or _get_package_os_type()
        self.doc_name = doc_name or "poptorch-user-guide"
        self.project_name = project_name or "poptorch"
        self.version_long = self.version
        self.poptorch_hash = _get_poptorch_hash()
        if build_number:
            self.version_long += "+" + build_number
        logger.debug("Adding custom attributes: %s", kwargs)
        self.__dict__.update(kwargs)
        logger.info("PkgInfo initialised: %s", str(self.__dict__))

    def pdf_filename(self):
        return f"{self.doc_name}-{self.version}-{self.snapshot}.pdf"

    def html_filename(self):
        return f"{self.doc_name}-html-{self.version}-{self.snapshot}.zip"

    def prodinfo_filename(self):
        return f"{self.project_name}-{self.version}-{self.snapshot}.yml"

    def save_to_file(self):
        with open(PkgInfo._pkg_info_file, "w") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def load_from_file(must_exist=False, path="."):
        pkg_info_path = os.path.join(path, PkgInfo._pkg_info_file)
        if not os.path.exists(pkg_info_path):
            if not must_exist:
                logger.info("Using default PkgInfo() options")
                return PkgInfo()
            raise FileNotFoundError(f"{pkg_info_path} not found")
        logger.info("Loading packaging options from %s", pkg_info_path)
        with open(pkg_info_path, "r") as f:
            attrs = json.load(f)
            return PkgInfo(**attrs)


def _get_version():
    v = json.load(open(os.path.join(sources_dir(), "version.json")))
    return f"{v['major']}.{v['minor']}.{v['point']}"


def _get_view_hash():
    try:
        hash = subprocess.check_output(
            [
                "git", "--git-dir",
                os.path.join(os.path.dirname(sources_dir()), ".git"),
                "rev-parse", "--short=10", "HEAD"
            ],
            stderr=subprocess.STDOUT).decode("utf-8").strip().rstrip()
        return hash
    except subprocess.CalledProcessError:
        return None


def _get_poptorch_hash():
    try:
        hash = subprocess.check_output(
            [
                "git", "--git-dir",
                os.path.join(sources_dir(), ".git"), "rev-parse", "--short=10",
                "HEAD"
            ],
            stderr=subprocess.STDOUT).decode("utf-8").strip().rstrip()
        return hash
    except subprocess.CalledProcessError:
        return None


def _get_snapshot():
    """ Use the view hash if available.
    Use the PopTorch hash as a fallback.
    Use 0000000000 if no git repository is found
    """
    snapshot = _get_view_hash()
    if snapshot:
        logger.debug("Using View hash %s as snapshot", snapshot)
        return snapshot
    snapshot = _get_poptorch_hash()
    if snapshot:
        logger.debug("Using PopTorch hash %s as snapshot", snapshot)
        return snapshot
    logger.debug("No git hash found to use as snapshot")
    return "0000000000"


def _get_package_os_type():
    distrib = None
    version = None
    for line in open("/etc/os-release", "r"):
        if line.startswith("ID="):
            distrib = line.split("=")[1].rstrip()
            distrib = distrib.replace('"', "")
        elif line.startswith("VERSION_ID="):
            version = line.split("=")[1]
            version = version.replace(".", "_")
            version = version.replace('"', "").rstrip()
    assert distrib and version
    return f"{distrib}_{version}"


def get_os_type():
    p = platform.uname()
    if p.system == "Darwin":
        return OsType.Osx
    if p.system == "Linux":
        return OsType.Linux

    return OsType.Unknown


def _make_output_non_blocking(output):
    fd = output.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    return output


class _LinesProcessor:
    def __init__(self, printer_fn):
        self.printer_fn = printer_fn
        self.partial_line = ""

    def _is_full_line(self, line):
        return line[-1] == "\n"

    def process(self, lines, flush=False):
        """ Due to buffering we need to check if lines
        are actual lines or just fragment of lines (in which case we
        wait until we've got the whole line available to print it).
        """
        for line in lines:
            self.partial_line += line.decode("utf-8")
            if self._is_full_line(self.partial_line):
                self.printer_fn(self.partial_line.rstrip())
                self.partial_line = ""
        if flush and self.partial_line:
            self.printer_fn(self.partial_line.rstrip())
            self.partial_line = ""


def run_commands(*commands, env=None, stop_on_error=True):
    bash_flags = ""
    if logger.isEnabledFor(logging.DEBUG):
        bash_flags += "x"  # print commands
    if stop_on_error:
        bash_flags += "e"

    if bash_flags:
        bash_flags = "set -" + bash_flags + ";"

    logger.debug("Running: %s", commands)
    p = subprocess.Popen([bash_flags + ";".join(commands)],
                         shell=True,
                         env=env,
                         executable='/bin/bash',
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    _make_output_non_blocking(p.stdout)
    _make_output_non_blocking(p.stderr)

    process_alive = True
    stdout = _LinesProcessor(logger.info)
    stderr = _LinesProcessor(logger.error)
    while process_alive:
        process_alive = p.poll() is None
        # Only flush if it is the last iteration
        stderr.process(p.stderr.readlines(), not process_alive)
        stdout.process(p.stdout.readlines(), not process_alive)

    assert p.returncode == 0, (f"{commands} failed with "
                               f"return code {p.returncode}")
