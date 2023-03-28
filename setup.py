# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pathlib
import os
import sys
import logging
from setuptools import setup
from setuptools.dist import Distribution
from pybind11.setup_helpers import Pybind11Extension

logging.basicConfig(level=logging.INFO)

# torch{audio, vision} are added here to prevent the torch upgrade when other
# packages depend on torch{audio, vision}.
REQUIRES = [
    'tqdm', '@TORCH_DEPENDENCY@', '@TORCHAUDIO_DEPENDENCY@',
    '@TORCHVISION_DEPENDENCY@'
]
VERSION = "@VERSION@"

LONG_DESCRIPTION = (
    "PopTorch is a set of extensions for PyTorch enabling "
    "models to be trained, evaluated and used on the Graphcore IPU.")

LIBS = ["*.so", "lib/*", "lib/poplar_rt/*", "lib/graphcore/lib/*.a"]


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


def get_torch_paths():
    # setup.py is executed several times, so it's ok if torch is not always
    # available.
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        return [], []
    torch_root = str(pathlib.Path(torch.__file__).parent)
    return [
        os.path.join(torch_root, "include"),
        os.path.join(torch_root, "include", "torch", "csrc", "api", "include")
    ], [os.path.join(torch_root, "lib")]


torch_include_dirs, torch_lib_dirs = get_torch_paths()
package_data = {'poptorch': LIBS}

# Copy custom codelets into the package so that we can pre-compile them later.
package_data["poptorch"].append("*.inc.cpp")

core_mod = Pybind11Extension(
    "poptorch.poptorch_core", ["src/poptorch.cpp"],
    define_macros=[("_GLIBCXX_USE_CXX11_ABI", 0)],
    include_dirs=["include"] + torch_include_dirs,
    library_dirs=["poptorch/lib"] + torch_lib_dirs,
    extra_link_args=["-Wl,--rpath=$ORIGIN/lib:$ORIGIN"],
    libraries=[
        "poptorch", "popart_compiler", "poptorch_err", "poptorch_logging",
        "torch_python", "torch"
    ],
    language="c++",
    cxx_std="17")

# Same as pybind11_add_module but without stripping the symbols and setting the visibility to hidden.
# Source: https://pybind11.readthedocs.io/en/stable/compiling.html#advanced-interface-library-targets
#
# If the symbols are stripped then error messages will only contain symbol
# addresses instead of human readable names.
core_mod.extra_compile_args = [
    f for f in core_mod.extra_compile_args
    if not "visibility=hidden" in f and not "-g0" in f
]

setup(
    name='poptorch',
    version=VERSION,
    description=LONG_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='http://graphcore.ai',
    author='Graphcore',
    author_email='contact@graphcore.ai',
    ext_modules=[core_mod],
    has_ext_modules=lambda: True,
    license='MIT License',
    license_files=('License.txt', 'poptorch_third_party_licenses.txt'),
    packages=['poptorch'],
    package_data=package_data,
    include_package_data=True,
    python_requires=f"=={sys.version_info.major}.{sys.version_info.minor}.*",
    platforms="@PLATFORM@",
    install_requires=REQUIRES,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
