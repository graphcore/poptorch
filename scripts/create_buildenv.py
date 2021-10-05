#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import contextlib
import fcntl
import hashlib
import inspect
import logging
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request

from utils import _utils

logger = logging.getLogger(os.path.basename(__file__))
_utils.set_logger(logger)

_conda_toolchains_packages = ["gcc_linux-64=7.3.0", "gxx_linux-64=7.3.0"]


def _default_cache_dir():
    return os.environ.get("CONDA_CACHE_DIR",
                          os.path.join(_utils.sources_dir(), ".cache"))


def _system_conda_path():
    try:
        conda_root = subprocess.check_output(["conda", "info", "--base"],
                                             stderr=None)
        conda_root = conda_root.decode("utf-8").strip()
        return conda_root
    except FileNotFoundError:
        return None


class Installer:
    """Common interface for all installers"""

    def install(self, env):
        raise Exception("Must be implemented by child class")

    def __hash__(self):
        raise Exception("Must be implemented by child class")


class CondaPackages(Installer):
    """Install the list of Conda packages in the environment."""

    def __init__(self, *packages):
        assert all([isinstance(p, str) for p in packages])
        self.packages = packages

    def __hash__(self):
        return hash(self.packages)


class PipPackages(Installer):
    """Install the list of pip3 packages in the environment."""

    def __init__(self, *packages):
        assert all([isinstance(p, str) for p in packages])
        self.packages = packages

    def install(self, env):
        env.run_commands("pip3 install " + " ".join(self.packages))

    def __hash__(self):
        return hash(self.packages)


class PipRequirements(Installer):
    """Install pip3 packages from a requirements file."""

    def __init__(self, filename="requirements.txt"):
        if not filename.startswith("/"):
            filename = os.path.join(os.getcwd(), filename)
        self._requirements_file = filename

    def install(self, env):
        env.run_commands(f"pip3 install -r {self._requirements_file}")

    def __hash__(self):
        with open(self._requirements_file, "r") as f:
            return hash(f.read())


class Installers:
    """Contains the list of installers to install in the environment."""

    def __init__(self):
        self._installers = []

    def add(self, installer):
        assert isinstance(
            installer,
            Installer), "All package installers must inherit from Installer"
        self._installers.append(installer)

    def __call__(self):
        return self._installers


class Config:
    """Contains the configuration for the environment."""

    def __init__(self, **opts):
        self.__dict__ = opts


class Environment:
    def __init__(self, buildenv_dir, activate_filename):
        self._buildenv_dir = buildenv_dir
        self._activate_filename = activate_filename

    @property
    def prefix(self):
        return self._buildenv_dir

    def run_commands(self,
                     *cmds,
                     env=None,
                     stop_on_error=True,
                     stdout_handler=None,
                     stderr_handler=None):
        _utils.run_commands(f". {self._activate_filename}",
                            *cmds,
                            env=env,
                            stop_on_error=stop_on_error,
                            stdout_handler=stdout_handler,
                            stderr_handler=stderr_handler)

    def rmdir_if_exists(self, path):
        _utils.rmdir_if_exists(path)


class BuildenvManager:
    def __init__(self,
                 cache_dir=None,
                 output_dir=None,
                 python_version=None,
                 use_conda_toolchains=False,
                 install_linters=False,
                 **config):
        python_version = python_version or platform.python_version()
        self.output_dir = os.path.realpath(output_dir or os.getcwd())
        self.cache_dir = cache_dir or _default_cache_dir()
        self.buildenv_dir = os.path.join(self.output_dir, "buildenv")
        self.conda_packages = [
            f"python={python_version}", "gdb=8.3", "conda-pack=0.5.0"
        ]
        self.projects = {}

        if use_conda_toolchains:
            self.conda_packages += _conda_toolchains_packages

        self.config = Config(install_linters=install_linters, **config)
        assert self.output_dir != _utils.sources_dir(), (
            "This script needs "
            "to be called from a build directory. Try mkdir build && cd build"
            " && ../scripts/create_buildenv.py")

        # internal constants
        self.activate_filename = os.path.join(self.output_dir,
                                              "activate_buildenv.sh")
        self.env = Environment(self.buildenv_dir, self.activate_filename)
        self.lock_already_acquired = False

    def add_project(self, project, project_dir):
        assert os.path.exists(project_dir)
        self.projects[project] = os.path.realpath(project_dir)

    def _collect_installers(self):
        view_dir = os.path.dirname(_utils.sources_dir())
        installers = Installers()
        # We share with the config files all the classes inheriting from Installer
        exec_locals = {
            name: c
            for name, c in inspect.getmembers(sys.modules[__name__],
                                              inspect.isclass)
            if Installer in c.__bases__ or c == Installer
        }
        exec_locals["installers"] = installers
        exec_locals["config"] = self.config
        for p, project_dir in self.projects.items():
            # Try to find (in that order):
            # 1) <view_dir>/my_project.buildenv.py
            # 2) <view_dir>/my_project/config.buildenv.py
            to_test = [
                os.path.join(view_dir, p + ".buildenv.py"),
                os.path.join(project_dir, "config.buildenv.py")
            ]
            conf = None
            for f in to_test:
                if os.path.exists(f):
                    conf = f
                    break
            if conf is None:
                logger.warning(
                    "No requirements found for project '%s' (Tried %s)", p,
                    to_test)
                continue

            with open(conf, "r") as f:
                code = f.read()
                os.chdir(project_dir)
                # Share the os module as it's commonly used to get the current
                # working directory, create directories, etc.
                # pylint: disable=exec-used
                exec(code, {"os": os}, exec_locals)

        # Process the installers:
        other_installers = []
        for i in installers():
            if isinstance(i, CondaPackages):
                self.conda_packages += i.packages
            else:
                other_installers.append(i)

        # Make sure the packages are unique and in a deterministic order
        self.conda_packages = list(sorted(dict.fromkeys(self.conda_packages)))

        return other_installers

    def create(self, create_template_if_needed=False):
        os.makedirs(self.output_dir, exist_ok=True)
        os.chdir(self.output_dir)

        self._clear_activate_buildenv()
        self._install_conda_if_needed()

        installers = self._collect_installers()
        env_hash = self._compute_environment_hash(installers)
        template_name = f"poptorch_{env_hash}.tar.gz"
        full_template_name = os.path.join(self.cache_dir, template_name)

        with self.cache_lock():
            if os.path.isfile(full_template_name):
                logger.info("Found template %s: Unpacking to %s",
                            full_template_name, self.buildenv_dir)
                os.makedirs(self.buildenv_dir)
                os.chdir(self.output_dir)
                tar = tarfile.open(full_template_name)
                tar.extractall(self.buildenv_dir)
                assert os.path.isdir(self.buildenv_dir)
                self.env.run_commands(f". {self.buildenv_dir}/bin/activate",
                                      "conda-unpack")
                self._append_to_activate_buildenv(
                    f"conda activate {self.buildenv_dir}", )
            else:
                logger.info(
                    "Didn't find template %s: creating a new "
                    "environment in %s", full_template_name, self.output_dir)
                self._create_new_env(installers)
                if create_template_if_needed:
                    os.chdir(self.output_dir)
                    self.env.run_commands(
                        f"conda activate {self.buildenv_dir}",
                        f"conda pack -p {self.buildenv_dir} -o \
                                {full_template_name}")

        os.chdir(self.output_dir)
        self.env.run_commands(
            """echo "export CCACHE_CPP2=yes" >> %s""" % self.activate_filename,
            """echo "export CC=\\"ccache ${CC:-gcc}\\"" >> %s""" %
            self.activate_filename,
            """echo "export CXX=\\"ccache ${CXX:-g++}\\"" >> %s""" %
            self.activate_filename)

    def _create_new_env(self, installers, is_retry=False):
        """
        Sometimes the Conda install in the NFS cache gets corrupted:

            CondaVerificationError: The package for setuptools located at
            /nfs/conda//miniconda/pkgs/setuptools-58.0.4-py38h578d9bd_2
            appears to be corrupted.

        When this happens: delete the conda install and start again with
        "is_retry=True" to avoid getting stuck in an infinite loop.
        """

        os.chdir(self.output_dir)
        corrupted = False

        def check_corruption(line):
            nonlocal corrupted
            if "CondaVerificationError" in line:
                corrupted = True
            logger.error(line)

        stderr_handler = None if is_retry else check_corruption
        try:
            _utils.rmdir_if_exists(self.buildenv_dir)
            self.env.run_commands(
                f"conda create --prefix {self.buildenv_dir} -c conda-forge "
                f"-y {' '.join(self.conda_packages)}",
                stderr_handler=stderr_handler)
        except AssertionError:
            if corrupted:
                # We failed because of some corrupted packages: clear
                # the environment, reinstall Conda and try again.
                self._clear_activate_buildenv()
                self._install_conda_if_needed(force_reinstall=True)
                self._create_new_env(installers, is_retry=True)
                return
            raise

        self._append_to_activate_buildenv(
            f"conda activate {self.buildenv_dir}", )

        for i in installers:
            os.chdir(self.output_dir)
            i.install(self.env)

    def _clear_activate_buildenv(self):
        open(self.activate_filename, "w").close()

    def _append_to_activate_buildenv(self, *lines):
        with open(self.activate_filename, "a") as f:
            for line in lines:
                f.write(f"{line}\n")

    @contextlib.contextmanager
    def cache_lock(self):
        # Handle nested cache_lock scopes: if we already own the lock then
        # don't try to lock it again.
        if self.lock_already_acquired:
            yield
            return

        lock = os.path.join(self.cache_dir, "conda.lock")
        with open(lock, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                self.lock_already_acquired = True
                yield
            finally:
                self.lock_already_acquired = False
                fcntl.flock(f, fcntl.LOCK_UN)

    def _install_conda_if_needed(self, force_reinstall=False):
        os.makedirs(self.cache_dir, exist_ok=True)
        system_conda = _system_conda_path()
        if system_conda is not None:
            logger.info("Using system conda")
            conda_sh = os.path.join(system_conda, "etc", "profile.d",
                                    "conda.sh")
            self._append_to_activate_buildenv(f". {conda_sh}")
            return

        miniconda_install_dir = os.path.join(self.cache_dir, "miniconda")
        conda_sh = os.path.join(miniconda_install_dir, "etc", "profile.d",
                                "conda.sh")
        installer = os.path.join(self.cache_dir, "Miniconda_installer.sh")
        with self.cache_lock():
            if os.path.isfile(conda_sh) and not force_reinstall:
                logger.info(
                    "System conda not found, using the instance from the cache "
                    "(%s) instead", self.cache_dir)
            else:
                logger.info(
                    "System conda not found, installing it locally in (%s)",
                    self.cache_dir)
                if not os.path.isfile(installer):
                    logger.info("Installer not found: downloading...")
                    conda_os = ""
                    os_type = _utils.get_os_type()
                    if os_type == _utils.OsType.Linux:
                        conda_os = "Linux"
                    elif os_type == _utils.OsType.Osx:
                        conda_os = "MacOSX"
                    else:
                        raise RuntimeError(
                            "Unknown OS. Please download the "
                            "installer for your platform from "
                            "https://repo.anaconda.com/miniconda/ and save it "
                            f"as ${installer}")
                    url = f"https://repo.anaconda.com/miniconda/Miniconda3-latest-{conda_os}-x86_64.sh"
                    urllib.request.urlretrieve(url, installer)
                _utils.rmdir_if_exists(miniconda_install_dir)
                _utils.run_commands(
                    f"bash {installer} -b -p {miniconda_install_dir}")
        assert os.path.isfile(conda_sh)
        self._append_to_activate_buildenv(f". {conda_sh}")

    def _compute_environment_hash(self, installers):
        hashes = [str(hash(i)) for i in installers]
        return str(
            hashlib.md5(" ".join(self.conda_packages +
                                 hashes).encode("utf-8")).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
                        "-d",
                        action="store_true",
                        help="Print debug messages")
    parser.add_argument(
        "--conda-toolchains",
        "-t",
        action="store_true",
        help="Use Conda toolchains instead of the system ones.")
    parser.add_argument("--popart-deps",
                        action="store_true",
                        help="Install dependencies to build PopART.")
    parser.add_argument("--no-linters",
                        action="store_true",
                        help="Don't install the linters.")
    parser.add_argument(
        "--python-version",
        "-p",
        help="Override the default python version used in the build environment"
        "By default the build environment will use the same python version as "
        "the host os")
    parser.add_argument(
        "--cache-dir",
        help=f"Cache directory (By default {_default_cache_dir()}")
    parser.add_argument(
        "--output-dir",
        help=
        "Where to create the build environment (Current directory by default)")
    parser.add_argument(
        "--create-template-if-needed",
        action="store_true",
        help="Create a template archive in the cache directory "
        "if one doesn't already exist")

    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    logger.debug("Args: %s", str(args))

    manager = BuildenvManager(args.cache_dir, args.output_dir,
                              args.python_version, args.conda_toolchains,
                              not args.no_linters)
    manager.add_project("poptorch", _utils.sources_dir())
    manager.create(args.create_template_if_needed)
