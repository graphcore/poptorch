#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# realpath doesn't exist on osx
realpath() {
  python3 -c "import os.path; print(os.path.realpath('$1'))"
}
set -e # Stop on error
SRC=$(realpath $(dirname $0))
if [ ${SRC} = $(realpath $(pwd)) ]
then
  echo "Warning: This script needs to be called from a build directory"
  echo "Trying to use ../build"
  mkdir ../build
  cd ../build
fi
BUILD=$(pwd)
if ! command -v conda &> /dev/null
then
  echo "System conda not found: installing locally"
  INSTALLER=${SRC}/Miniconda_installer.sh
  if [ ! -f ${INSTALLER} ]
  then
    echo "Installer not found: downloading..."
    case "$(uname -s)" in
      Darwin*)
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ${INSTALLER}
        ;;
      Linux*)
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ${INSTALLER}
        ;;
      *)
        echo "Unknown platform: $(uname -s). Please download the installer for your platform from https://repo.anaconda.com/miniconda/ and save it as ${INSTALLER}"
        exit -1
        ;;
    esac
  fi
  rm -rf miniconda
  bash ${INSTALLER} -b -p miniconda
  echo ". `pwd`/miniconda/etc/profile.d/conda.sh" > $BUILD/activate_buildenv.sh
  . $BUILD/activate_buildenv.sh
fi
cd $BUILD
conda create --prefix $BUILD/buildenv -c conda-forge -y python=3.6.9 pybind11 pytest spdlog=1.8.0 ninja cmake=3.18.2 protobuf=3.7.1 latexmk zip make
conda activate $BUILD/buildenv
echo "conda activate $BUILD/buildenv" >> $BUILD/activate_buildenv.sh
pip install -r ${SRC}/requirements.txt

# Build onnx for now (Waiting for https://github.com/conda-forge/staged-recipes/pull/13108 to be merged)
mkdir $BUILD/onnx
cd $BUILD/onnx
curl -L https://github.com/onnx/onnx/archive/v1.7.0.tar.gz | tar zx --strip-components=1
mkdir build && cd build
cmake ../ -GNinja -DONNX_ML=0 -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
ninja install
