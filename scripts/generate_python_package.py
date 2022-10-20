#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import datetime
import os
import tempfile
import subprocess
import shutil
import distutils.util
import distutils.dir_util
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
import utils._utils as utils

targets = ["bdist_wheel", "sdist", "install"]

parser = argparse.ArgumentParser()
parser.add_argument("--python-dir",
                    default="include",
                    help="Path to the folder containing the python files")
parser.add_argument(
    "--include-dir",
    default="include",
    help="Path to the include folder needed to compile the wheel")
parser.add_argument(
    "--lib-dir",
    default="lib",
    help=
    "Path to the folder containing the libraries needed to compile the wheel")
parser.add_argument(
    "--standalone",
    help=("Colon separated list of folders to add to the lib folder of the "
          "sdist / wheel package"))
parser.add_argument("target",
                    choices=targets,
                    help=f"Which target to build: {targets}")
parser.add_argument("--output-dir",
                    default="dist",
                    help="Where to create the packages")
parser.add_argument("--build-without-mlir",
                    action="store_true",
                    help="Build the python bindings without MLIR support")
args = parser.parse_args()


def get_torch_dependency():
    if "aarch64" in utils.get_arch_type():
        # There is no +cpu variant of Torch on Arm
        return f'torch=={TORCH_VERSION}'
    # For sdist packages we don't know ahead of time what the python version
    # will be, and there is no support for --find-links so we just have to
    # use the regular torch wheel instead.
    if args.target != "bdist_wheel":
        return f'torch=={TORCH_VERSION}'
    return f"torch @ https://download.pytorch.org/whl/cpu/torch-{TORCH_VERSION}%2Bcpu-{get_abbr_impl()}{get_impl_ver()}-{get_abi_tag()}-{PLATFORM}.whl"


def get_poptorch_version():
    version = utils.PkgInfo.load_from_file(must_exist=False,
                                           path="..").version_long
    if args.standalone is not None:
        # Only 1 "+" symbol allowed per version
        separator = "+" if "+" not in version else "_"
        version += separator + "standalone"
    return version


def get_define_macros():
    macros = "POPTORCH_BUILD_MLIR_COMPILER="
    if args.build_without_mlir:
        macros += "0"
    else:
        macros += "1"
    return macros


def update_ldshared():
    """Only needed on CentOS 7."""
    try:
        with open("/etc/redhat-release") as f:
            content = f.read()
            if "CentOS" in content and " 7." in content:
                return "True"
    except FileNotFoundError:
        pass
    return "False"


VERSION = get_poptorch_version()
TORCH_VERSION = utils.get_required_torch_version()
UPDATE_LDSHARED = update_ldshared()
DEFINE_MACROS = get_define_macros()

# https://www.python.org/dev/peps/pep-0425/
# The platform tag is simply distutils.util.get_platform() with all hyphens - and periods . replaced with underscore _.
PLATFORM = distutils.util.get_platform().replace(".", "_").replace("-", "_")

TORCH_DEPENDENCY = get_torch_dependency()


# Only keep files of a given extension
class ExtOnly:
    def __init__(self, *ext):
        self.ext = ext

    def _is_ignored(self, file):
        return not any(file.endswith(ext) for ext in self.ext)

    def __call__(self, adir, filenames):
        # Return the files to ignore
        return [f for f in filenames if self._is_ignored(f)]


src_dir = utils.sources_dir()
include_dir = os.path.realpath(args.include_dir)
lib_dirs = [os.path.realpath(args.lib_dir)]
if args.standalone is not None:
    lib_dirs += [os.path.realpath(l) for l in args.standalone.split(":")]
output_dir = os.path.realpath(args.output_dir)
python_dir = os.path.realpath(args.python_dir)


def configure(src_filename, dst_filename):
    with open(dst_filename, "w") as f:
        for line in open(src_filename):
            f.write(
                line.replace("@VERSION@", VERSION).replace(
                    "@TORCH_DEPENDENCY@",
                    TORCH_DEPENDENCY).replace("@PLATFORM@", PLATFORM).replace(
                        "@UPDATE_LDSHARED@",
                        UPDATE_LDSHARED).replace("@DEFINE_MACROS@",
                                                 DEFINE_MACROS))


# Create a temporary directory and copy the files to package to it.
with tempfile.TemporaryDirectory() as tmp_dir:
    os.chdir(tmp_dir)
    shutil.copytree(os.path.join(src_dir, "python"),
                    "src",
                    ignore=ExtOnly(".cpp"))
    shutil.copytree(python_dir, "poptorch")
    # distutils won't throw an exception if the destination already exists,
    # which will happen if lib_dirs contains more than one element.
    for lib_dir in lib_dirs:
        distutils.dir_util.copy_tree(lib_dir, "poptorch/lib")
    shutil.copytree(include_dir, "include")
    shutil.copy(os.path.join(src_dir, "MANIFEST.in"), ".")

    configure(os.path.join(src_dir, "setup.py"), "setup.py")
    configure(os.path.join(src_dir, "pyproject.toml"), "pyproject.toml")

    # distutils doesn't like spaces in CXX (https://github.com/mapnik/python-mapnik/issues/99#issuecomment-527591113)
    env = {**os.environ}
    cc = env.get("CC", "gcc")
    cxx = env.get("CXX", "g++")
    # Only keep the real compiler: e.g "cmake gcc" -> "gcc"
    cc = cc.split(" ")[-1]
    cxx = cxx.split(" ")[-1]
    env["CXX"] = cxx
    env["CC"] = cc
    start = datetime.datetime.now()
    if args.target == "install":
        subprocess.check_call(
            f"python3 setup.py build_ext -b {output_dir}".split(), env=env)
        dst_dir = f"{output_dir}/poptorch/lib"
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree("poptorch/lib", dst_dir, ignore=ExtOnly(".so"))
    else:
        subprocess.check_call(
            f"python3 setup.py {args.target} -d {output_dir}".split(), env=env)
    print(f"Time to generate {args.target} in {output_dir} : "
          f"{datetime.datetime.now()-start}")
