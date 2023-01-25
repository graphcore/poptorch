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
                                       path='../poptorch').version_long

# https://www.python.org/dev/peps/pep-0425/
# The platform tag is simply distutils.util.get_platform() with all hyphens - and periods . replaced with underscore _.
PLATFORM = distutils.util.get_platform().replace('.', '_').replace('-', '_')


def find_requirement(package):
    with open(os.path.join(src_dir, 'requirements.txt'), 'r') as f:
        for line in f:
            if package in line:
                return line.strip()

    return None


PYG_DEPENDENCY = find_requirement('torch-geometric') or find_requirement(
    'pyg-nightly')

if PYG_DEPENDENCY is None:
    raise RuntimeError('"torch-geometric" not found in requirements.txt')

POPTORCH_DEPENDENCY = f'poptorch=={VERSION}'


def configure(src_filename, dst_filename):
    with open(dst_filename, 'w') as f:
        for line in open(src_filename):
            f.write(
                line.replace('@VERSION@', VERSION) \
                    .replace('@PYG_DEPENDENCY@', PYG_DEPENDENCY) \
                    .replace('@POPTORCH_DEPENDENCY@', POPTORCH_DEPENDENCY) \
                    .replace('@PLATFORM@', PLATFORM)
            )


# Create a temporary directory and copy the files to package to it.
with tempfile.TemporaryDirectory() as tmp_dir:
    os.chdir(tmp_dir)
    shutil.copytree(os.path.join(src_dir, 'python'), 'src')
    shutil.copytree(python_dir, PROJ_NAME)
    shutil.copy(os.path.join(src_dir, 'MANIFEST.in'), '.')

    configure(os.path.join(src_dir, 'setup.py'), 'setup.py')
    configure(os.path.join(src_dir, 'pyproject.toml'), 'pyproject.toml')

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
