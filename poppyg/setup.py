# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from setuptools import setup

REQUIRES = [
    '@PYG_DEPENDENCY@', '@POPTORCH_DEPENDENCY@', 'torch-scatter',
    'torch-sparse', 'torch-cluster', 'torch-spline-conv'
]
VERSION = "@VERSION@"

LONG_DESCRIPTION = (
    "PopPyG is a set of extensions for PyTorch Geometric, enabling "
    "GNN models to be trained, evaluated and used on the Graphcore IPU.")

setup(name='poppyg',
      version=VERSION,
      description=LONG_DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      license='Apache 2.0',
      author="Graphcore Ltd.",
      author_email='contact@graphcore.ai',
      url='http://graphcore.ai',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          'Intended Audience :: Science/Research',
          "Topic :: Scientific/Engineering",
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'License :: OSI Approved :: Apache Software License',
          "Programming Language :: Python :: 3",
      ],
      platforms="@PLATFORM@",
      install_requires=REQUIRES,
      python_requires=">=3.7",
      packages=['poppyg'])
