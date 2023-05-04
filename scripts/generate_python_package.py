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
args = parser.parse_args()


def get_version_from_requirements(package):
    with open(os.path.join(src_dir, 'requirements.txt'), 'r') as f:
        for line in f:
            if package in line and not 'cpu' in line:
                name_and_version = line.split(';')[0].split('=')
                return name_and_version[-1].strip()

    return None


def get_torch_dependency(package, version):
    if "aarch64" in utils.get_arch_type():
        # There is no +cpu variant of Torch on Arm
        return f'{package}=={version}'
    # For sdist packages we don't know ahead of time what the python version
    # will be, and there is no support for --find-links so we just have to
    # use the regular torch wheel instead.
    if args.target != "bdist_wheel":
        return f'{package}=={version}'
    return f"{package} @ https://download.pytorch.org/whl/cpu/{package}-{version}%2Bcpu-{get_abbr_impl()}{get_impl_ver()}-{get_abi_tag()}-{PLATFORM}.whl"


def get_poptorch_version():
    version = utils.PkgInfo.load_from_file(must_exist=False,
                                           path="..").version_long
    if args.standalone is not None:
        # Only 1 "+" symbol allowed per version
        separator = "+" if "+" not in version else "_"
        version += separator + "standalone"
    return version


VERSION = get_poptorch_version()

# https://www.python.org/dev/peps/pep-0425/
# The platform tag is simply distutils.util.get_platform() with all hyphens - and periods . replaced with underscore _.
PLATFORM = distutils.util.get_platform().replace(".", "_").replace("-", "_")

torch_ver = utils.get_required_torch_version()
TORCH_DEPENDENCY = get_torch_dependency('torch', torch_ver)

src_dir = utils.sources_dir()
# torch{audio, vision} are added here to prevent the torch upgrade when other
# packages depend on torch{audio, vision}.
torchaudio_ver = get_version_from_requirements('torchaudio')
TORCHAUDIO_DEPENDENCY = get_torch_dependency('torchaudio', torchaudio_ver)

torchvision_ver = get_version_from_requirements('torchvision')
TORCHVISION_DEPENDENCY = get_torch_dependency('torchvision', torchvision_ver)


# Only keep files of a given extension
class ExtOnly:
    def __init__(self, *ext):
        self.ext = ext

    def _is_ignored(self, file):
        return not any(file.endswith(ext) for ext in self.ext)

    def __call__(self, adir, filenames):
        # Return the files to ignore
        return [f for f in filenames if self._is_ignored(f)]


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
                line.replace("@VERSION@", VERSION) \
                    .replace("@PLATFORM@", PLATFORM) \
                    .replace("@TORCH_DEPENDENCY@", TORCH_DEPENDENCY) \
                    .replace("@TORCHAUDIO_DEPENDENCY@", TORCHAUDIO_DEPENDENCY) \
                    .replace("@TORCHVISION_DEPENDENCY@", TORCHVISION_DEPENDENCY)
            )


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
    shutil.copy(os.path.join(src_dir, 'setup.cfg'), '.')
    shutil.copy(os.path.join(src_dir, 'License.txt'), '.')
    shutil.copy(os.path.join(src_dir, 'poptorch_third_party_licenses.txt'),
                '.')

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
        extra_opts = ""
        if args.target == "sdist":
            extra_opts = "--formats=zip"
        subprocess.check_call(
            f"python3 setup.py {args.target} -d {output_dir} {extra_opts}".
            split(),
            env=env)
    print(f"Time to generate {args.target} in {output_dir} : "
          f"{datetime.datetime.now()-start}")
