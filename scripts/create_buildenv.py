#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import hashlib
import logging
import os
import platform
import subprocess
import tarfile
import urllib.request

from utils import _utils

logger = logging.getLogger(os.path.basename(__file__))
_utils.set_logger(logger)


def _default_cache_dir():
    return os.path.join(_utils.sources_dir(), ".cache")


def _system_conda_exists():
    try:
        subprocess.check_call(["conda", "--version"], stdout=None, stderr=None)
        return True
    except FileNotFoundError:
        return False


_conda_packages = [
    "ccache=3.7.9",
    "cmake=3.18.2",
    "conda-pack=0.5.0",
    "latexmk=4.55",
    "make=4.3",
    "ninja=1.10.2",
    "pybind11=2.6.1",
    "pytest=6.2.1",
    "zip=3.0",
    "spdlog=1.8.0",
    "wheel==0.34.2",
]
_protobuf_url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protobuf-python-3.14.0.tar.gz"
_onnx_url = "https://github.com/onnx/onnx/archive/v1.7.0.tar.gz"
_pip_requirements_file = os.path.join(_utils.sources_dir(), "requirements.txt")


class BuildenvManager:
    def __init__(self,
                 cache_dir=None,
                 output_dir=None,
                 python_version=None,
                 use_conda_toolchains=False,
                 build_protobuf=False):
        self.python_version = python_version or platform.python_version()
        self.use_conda_toolchains = use_conda_toolchains
        self.build_protobuf = build_protobuf
        self.output_dir = os.path.realpath(output_dir or os.getcwd())
        self.cache_dir = cache_dir or _default_cache_dir()
        self.buildenv_dir = os.path.join(self.output_dir, "buildenv")

        assert self.output_dir != _utils.sources_dir(), (
            "This script needs "
            "to be called from a build directory. Try mkdir build && cd build"
            " && ../scripts/create_buildenv.py")

        # internal constants
        self.activate_filename = "activate_buildenv.sh"

    def create(self, create_template_if_needed=False):
        os.makedirs(self.output_dir, exist_ok=True)
        os.chdir(self.output_dir)

        self._clear_activate_buildenv()
        self._install_conda_if_needed()

        env_hash = self._compute_environment_hash()
        template_name = f"poptorch_{env_hash}.tar.gz"
        full_template_name = os.path.join(self.cache_dir, template_name)

        if os.path.isfile(full_template_name):
            logger.info("Found template %s: Unpacking to %s",
                        full_template_name, self.buildenv_dir)
            os.makedirs(self.buildenv_dir)
            os.chdir(self.output_dir)
            tar = tarfile.open(full_template_name)
            tar.extractall(self.buildenv_dir)
            assert os.path.isdir(self.buildenv_dir)
            _utils.run_commands(f". {self.activate_filename}",
                                f". {self.buildenv_dir}/bin/activate",
                                "conda-unpack")
            self._append_to_activate_buildenv(
                f"conda activate {self.buildenv_dir}", )
        else:
            logger.info(
                "Didn't find template %s: creating a new "
                "environment in %s", full_template_name, self.output_dir)
            self._create_new_env()
            if create_template_if_needed:
                # TODO Create / check lock
                os.chdir(self.output_dir)
                _utils.run_commands(
                    f". {self.activate_filename}",
                    f"conda activate {self.buildenv_dir}",
                    f"conda pack -p {self.buildenv_dir} -o {full_template_name}"
                )

        os.chdir(self.output_dir)
        _utils.run_commands(
            f". {self.activate_filename}",
            """echo "export CCACHE_CPP2=yes" >> %s""" % self.activate_filename,
            """echo "export CC=\\"ccache ${CC:-gcc}\\"" >> %s""" %
            self.activate_filename,
            """echo "export CXX=\\"ccache ${CXX:-g++}\\"" >> %s""" %
            self.activate_filename)

    def _build_onnx(self):
        os.chdir(self.output_dir)
        os.makedirs(os.path.join("onnx", "build"))
        _utils.run_commands(
            f". {self.activate_filename}",
            "cd onnx",
            f"curl -sSL {_onnx_url} | tar zx --strip-components=1",
            "cd build",
            "cmake ../ -GNinja -DONNX_ML=0 \
                -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
                -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "ninja install",
        )
        _utils.rmdir_if_exists("onnx")

    def _build_protobuf(self):
        os.chdir(self.output_dir)
        os.makedirs("protobuf")
        _utils.run_commands(
            f". {self.activate_filename}",
            "cd protobuf",
            f"curl -sSL {_protobuf_url} | tar zx --strip-components=1",
            "CXXFLAGS=-fPIC CFLAGS=-fPIC ./configure --prefix=${CONDA_PREFIX}",
            "make -j`nproc`",
            "make install",
        )
        _utils.rmdir_if_exists("protobuf")

    def _create_new_env(self):
        os.chdir(self.output_dir)
        _utils.run_commands(
            f". {self.activate_filename}",
            f"conda create --prefix {self.buildenv_dir} -c conda-forge "
            f"-y {self._get_conda_packages_list()}")

        self._append_to_activate_buildenv(
            f"conda activate {self.buildenv_dir}", )

        _utils.run_commands(f". {self.activate_filename}",
                            f"pip3 install -r {_pip_requirements_file}")

        if self.build_protobuf:
            self._build_protobuf()

        self._build_onnx()

    def _clear_activate_buildenv(self):
        open(self.activate_filename, "w").close()

    def _append_to_activate_buildenv(self, *lines):
        with open(self.activate_filename, "a") as f:
            for line in lines:
                f.write(f"{line}\n")

    def _install_conda_if_needed(self):
        if _system_conda_exists():
            logger.info("Using system conda")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        miniconda_install_dir = os.path.join(self.cache_dir, "miniconda")
        conda_sh = os.path.join(miniconda_install_dir, "etc", "profile.d",
                                "conda.sh")
        installer = os.path.join(self.cache_dir, "Miniconda_installer.sh")
        # FIXME: Some kind of lock here?
        if os.path.isfile(conda_sh):
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
            _utils.run_commands(
                f"bash {installer} -b -p {miniconda_install_dir}")
        assert os.path.isfile(conda_sh)
        self._append_to_activate_buildenv(f". {conda_sh}")

    def _pip_requirements(self):
        with open(_pip_requirements_file, "r") as f:
            return f.read()

    def _get_conda_packages_list(self):
        pkgs = _conda_packages[:]
        pkgs += [f"python={self.python_version}"]
        if self.use_conda_toolchains:
            pkgs += ["gcc_linux-64=7.3.0", "gxx_linux-64=7.3.0"]
        if not self.build_protobuf:
            pkgs += ["protobuf=3.14.0"]
        return " ".join(pkgs)

    def _compute_environment_hash(self):
        protobuf_used = ""
        if self.build_protobuf:
            protobuf_used = _protobuf_url
        return str(
            hashlib.md5(
                (self._pip_requirements() + self._get_conda_packages_list() +
                 protobuf_used + _onnx_url).encode("utf-8")).hexdigest())


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
    parser.add_argument(
        "--build-protobuf",
        action="store_true",
        help="Build protobuf from sources instead of using the Conda package.")
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
                              args.build_protobuf)
    manager.create(args.create_template_if_needed)

    #FIXME: Clean up conda cache
