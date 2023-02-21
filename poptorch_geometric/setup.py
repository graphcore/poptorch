# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import sys

from setuptools import setup, find_packages

REQUIRES = [
    '@PYG_DEPENDENCY@',
    '@POPTORCH_DEPENDENCY@',
    '@TORCH_SCATTER_DEPENDENCY@',
    '@TORCH_SPARSE_DEPENDENCY@',
]
VERSION = '@VERSION@'

LONG_DESCRIPTION = (
    'PopTorch Geometric is a set of extensions for PyTorch Geometric, enabling '
    'GNN models to be trained, evaluated and used on the Graphcore IPU.')

setup(name='poptorch_geometric',
      version=VERSION,
      description=LONG_DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      license='MIT License',
      author='Graphcore Ltd.',
      author_email='contact@graphcore.ai',
      url='http://graphcore.ai',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      platforms='@PLATFORM@',
      install_requires=REQUIRES,
      python_requires=f'=={sys.version_info.major}.{sys.version_info.minor}.*',
      packages=find_packages())
