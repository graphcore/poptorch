# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import sys
import distutils
import distutils.util
import logging
from setuptools import setup
from setuptools.dist import Distribution
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
import scripts.utils._utils as utils

logging.basicConfig(level=logging.INFO)
VERSION = utils.PkgInfo.load_from_file(must_exist=False,
                                       path="..").version_long
TORCH_VERSION = utils.get_required_torch_version()

LONG_DESCRIPTION = (
    "PopTorch is a set of extensions for PyTorch enabling "
    "models to be trained, evaluated and used on the GraphCore IPU.")

# https://www.python.org/dev/peps/pep-0425/
# The platform tag is simply distutils.util.get_platform() with all hyphens - and periods . replaced with underscore _.
platform = distutils.util.get_platform().replace(".", "_").replace("-", "_")

if "macosx" in platform:
    REQUIRES = [f'torch=={TORCH_VERSION}']
    LIBS = ["*.dylib", "*.so"]
else:
    REQUIRES = [
        f'torch @ https://download.pytorch.org/whl/cpu/torch-{TORCH_VERSION}%2Bcpu-{get_abbr_impl()}{get_impl_ver()}-{get_abi_tag()}-{platform}.whl',
    ]
    LIBS = ["*.so"]


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


package_data = {'poptorch': LIBS}
wheel_lib_dirs = os.environ.get("WHEEL_LIB_DIRS")
lib = "poptorch/lib"
utils.rmdir_if_exists(lib)
if wheel_lib_dirs:
    os.makedirs(lib)
    for d in wheel_lib_dirs.split(":"):
        distutils.dir_util.copy_tree(d, lib)
    package_data["poptorch"].append("lib/*")
    package_data["poptorch"].append("lib/poplar_rt/*")
    package_data["poptorch"].append("lib/graphcore/lib/*.a")
    VERSION += "+standalone"

setup(
    name='poptorch',
    version=VERSION,
    description=LONG_DESCRIPTION[0],
    long_description=LONG_DESCRIPTION[3:],
    long_description_content_type="text/markdown",
    url='http://graphcore.ai',
    author='GraphCore',
    author_email='contact@graphcore.ai',
    license='Apache 2.0',
    packages=['poptorch'],
    package_data=package_data,
    python_requires=f"=={sys.version_info.major}.{sys.version_info.minor}.*",
    platforms=platform,
    install_requires=REQUIRES,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
# Don't keep the Poplar / Popart libs around
# This might interfere with the enable.sh script
utils.rmdir_if_exists(lib)
