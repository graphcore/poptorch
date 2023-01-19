# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os

# IMPORTANT: Keep requirements in sync with ./requirements.txt.


def find_pyg():
    pyg_names = ['pyg-nightly', 'torch-geometric']
    req_file = os.path.join(os.path.realpath(__file__), 'requirements.txt')
    with open(req_file, 'r') as f:
        for line in f:
            if any(package in line for package in pyg_names):
                line.replace('==', '=')
                return line.strip()

    RuntimeError('"torch-geometric" not found in requirements.txt')


installers.add(
    CondaPackages(find_pyg(), 'torch-scatter', 'torch-sparse', 'torch-cluster',
                  'torch-spline-conv', 'nbformat', 'nbconvert',
                  'pytest-benchmark', 'pytest-cov'))

installers.add(PipRequirements("requirements.txt"))
