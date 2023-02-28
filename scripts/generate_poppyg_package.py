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
import utils._utils as utils
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag

targets = ['bdist_wheel', 'sdist', 'install']

parser = argparse.ArgumentParser()
parser.add_argument('--python-dir',
                    help='Path to the folder containing the python files')
parser.add_argument('target',
                    choices=targets,
                    help=f'Which target to build: {targets}')
parser.add_argument('--output-dir',
                    default='dist',
                    help='Where to create the packages')
args = parser.parse_args()

PROJ_NAME = 'poptorch_geometric'
src_dir = os.path.join(utils.sources_dir(), PROJ_NAME)
output_dir = os.path.realpath(args.output_dir)
python_dir = os.path.realpath(args.python_dir)

VERSION = utils.PkgInfo.load_from_file(must_exist=False,
                                       path='..').version_long

# https://www.python.org/dev/peps/pep-0425/
# The platform tag is simply distutils.util.get_platform() with all hyphens - and periods . replaced with underscore _.
PLATFORM = distutils.util.get_platform().replace('.', '_').replace('-', '_')


def find_requirement(package):
    with open(os.path.join(src_dir, 'requirements.txt'), 'r') as f:
        for line in f:
            if package in line:
                return line.strip()

    return None


def get_pyg_hosted_dependency(pkg_name):
    name_and_version = find_requirement(pkg_name)
    assert name_and_version is not None, f'{pkg_name} not found.'

    # For sdist packages we don't know ahead of time what the python version
    # will be, and there is no support for --find-links so we just have to
    # use the regular wheel instead.
    if args.target != "bdist_wheel":
        return name_and_version

    pkg_ver = name_and_version.split('=')[-1]
    file_name = pkg_name.replace('-', '_')
    pkg_whl = f'{pkg_name} @ https://data.pyg.org/whl/torch-1.13.0%2Bcpu/{file_name}-{pkg_ver}-{get_abbr_impl()}{get_impl_ver()}-{get_abi_tag()}-{PLATFORM}.whl'

    return pkg_whl


PYG_DEPENDENCY = find_requirement('torch-geometric') or find_requirement(
    'pyg-nightly')

if PYG_DEPENDENCY is None:
    raise RuntimeError('"torch-geometric" not found in requirements.txt')

SCATTER_DEPENDENCY = get_pyg_hosted_dependency('torch-scatter')
SPARSE_DEPENDENCY = get_pyg_hosted_dependency('torch-sparse')

POPTORCH_DEPENDENCY = f'poptorch=={VERSION}'


def configure(src_filename, dst_filename):
    with open(dst_filename, 'w') as f:
        for line in open(src_filename):
            f.write(
                line.replace('@VERSION@', VERSION) \
                    .replace('@PYG_DEPENDENCY@', PYG_DEPENDENCY) \
                    .replace('@POPTORCH_DEPENDENCY@', POPTORCH_DEPENDENCY) \
                    .replace('@PLATFORM@', PLATFORM) \
                    .replace('@TORCH_SCATTER_DEPENDENCY@', SCATTER_DEPENDENCY) \
                    .replace('@TORCH_SPARSE_DEPENDENCY@', SPARSE_DEPENDENCY)
            )


# Create a temporary directory and copy the files to package to it.
with tempfile.TemporaryDirectory() as tmp_dir:
    os.chdir(tmp_dir)
    shutil.copytree(python_dir, PROJ_NAME)
    shutil.copy(os.path.join(src_dir, 'MANIFEST.in'), '.')
    shutil.copy(os.path.join(src_dir, 'License.txt'), '.')
    shutil.copy(os.path.join(src_dir, 'setup.cfg'), '.')

    configure(os.path.join(src_dir, 'setup.py'), 'setup.py')

    env = {**os.environ}
    start = datetime.datetime.now()

    if args.target == 'install':
        subprocess.check_call(
            f'python3 setup.py build_ext -b {output_dir}'.split(), env=env)
    else:
        extra_opts = ''
        if args.target == 'sdist':
            extra_opts = '--formats=zip'
        subprocess.check_call(
            f'python3 setup.py {args.target} -d {output_dir} {extra_opts}'.
            split(),
            env=env)

    print(f'Time to generate {args.target} in {output_dir} : '
          f'{datetime.datetime.now()-start}')
