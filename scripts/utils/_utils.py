# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import re
import enum
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
    def __init__(self,
                 version=None,
                 snapshot=None,
                 os_type=None,
                 package_os_type=None,
                 build_number=None,
                 doc_name=None,
                 project_name=None):
        logger.debug(
            "PkgInfo: user provided version=%s snapshot=%s os_type=%s"
            " package_os_type=%s build_number=%s doc_name=%s "
            "project_name=%s", version, snapshot, os_type, package_os_type,
            build_number, doc_name, project_name)
        self.version = version or _get_version()
        self.snapshot = snapshot or _get_snapshot()
        self.os_type = os_type or _get_os_type()
        if isinstance(self.os_type, OsType):
            self.os_type = self.os_type.value
        self.package_os_type = package_os_type or _get_package_os_type()
        self.name = doc_name or "poptorch-user-guide"
        self.project_name = project_name or "poptorch"
        self.version_long = self.version
        if build_number:
            self.version_long += "+" + build_number
        logger.debug("PkgInfo initialised: %s", str(self.__dict__))

    def pdf_filename(self):
        return f"{self.name}-{self.version}-{self.snapshot}.pdf"

    def html_filename(self):
        return f"{self.name}-html-{self.version}-{self.snapshot}.zip"

    def prodinfo_filename(self):
        return f"{self.project_name}-{self.version}-{self.snapshot}.yml"


class JenkinsPkgInfo(PkgInfo):
    """Version of PkgInfo used by the CI: gets override for default values
    from environment variables."""

    def __init__(self):
        build_number = os.environ.get("GC_BUILD_NUMBER")
        logger.debug("Env: GC_BUILD_NUMBER=%s", build_number)
        os_type = os.environ.get("GCCI_OS")
        os_version = os.environ.get("GCCI_OS_VERSION")

        logger.debug("Env: GCCI_OS=%s GCCI_OS_VERSION=%s", os_type, os_version)
        package_os_type = None
        if os_type and os_version:
            os_version = os_version.replace(".", "_")
            package_os_type = f"{os_type}_{os_version}"
            logger.info(
                "Package OS type set by 'GCCI_OS' / 'GCCI_OS_VERSION' "
                "to '%s'", package_os_type)
        super().__init__(build_number=build_number,
                         package_os_type=package_os_type)


def _get_version():
    v = json.load(open(os.path.join(sources_dir(), "version.json")))
    return f"{v['major']}.{v['minor']}.{v['point']}"


def _get_snapshot():
    try:
        return subprocess.check_output(
            [
                "git", "--git-dir",
                os.path.join(sources_dir(), ".git"), "rev-parse", "--short=10",
                "HEAD"
            ],
            stderr=subprocess.STDOUT).decode("utf-8").strip().rstrip()
    except subprocess.CalledProcessError:
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


def _get_os_type():
    p = platform.uname()
    if p.system == "Darwin":
        return OsType.Osx
    if p.system == "Linux":
        return OsType.Linux

    return OsType.Unknown
