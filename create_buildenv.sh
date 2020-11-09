#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# realpath doesn't exist on osx
realpath() {
  python3 -c "import os.path; print(os.path.realpath('$1'))"
}
set -e # Stop on error
SRC=$(realpath $(dirname $0))
CACHE=${SRC}/.cache
print_usage_and_exit() {
  cat << EOM
Usage: $0 [--create-template|-c] [--help|-h]

  -h, --help
    Print this help message
  -c, --create-template
    Create a poptorch template environment

If no argument is provided: create a poptorch build environment.
If a template environment matching the required config exists: clone it instead of creating an environment from scratch.

EOM
  exit 1
}

find_env() {
  # Don't let grep return 1 in case of nomatch
  conda env list | grep $1 || true
}

while [[ $# -gt 0 ]]
do
  case "$1" in
    -c|--create-template)
      CREATE_TEMPLATE=true
      shift
      ;;
    -h|--help)
      print_usage_and_exit
      shift
      ;;
  *)
    echo "Unknown argument '$1'"
    print_usage_and_exit
    shift
    ;;
  esac
done

if [ ${SRC} = $(realpath $(pwd)) ]
then
  echo "Warning: This script needs to be called from a build directory"
  echo "Trying to use ../build"
  mkdir ../build
  cd ../build
fi
BUILD=$(pwd)
rm -f $BUILD/activate_buildenv.sh
if ! command -v conda &> /dev/null
then
  mkdir -p ${CACHE}
  if [ -f ${CACHE}/miniconda/etc/profile.d/conda.sh ]
  then
    echo "System conda not found: using the one from ${CACHE}"
  else
    echo "System conda not found: installing locally in ${CACHE}"
    INSTALLER=${CACHE}/Miniconda_installer.sh
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
    bash ${INSTALLER} -b -p ${CACHE}/miniconda
  fi

  echo ". ${CACHE}/miniconda/etc/profile.d/conda.sh" > $BUILD/activate_buildenv.sh
  . $BUILD/activate_buildenv.sh
fi

cd $BUILD
BUILDENV=$BUILD/buildenv
checksum=$(python3 -c "import hashlib,pathlib;print(hashlib.md5(pathlib.Path('${SRC}/requirements.txt').read_bytes()+pathlib.Path('${SRC}/create_buildenv.sh').read_bytes()).hexdigest())")

template_name="${CACHE}/poptorch_${checksum}.tar.gz"

if [ ! -f "${template_name}" ]
then
  echo "Didn't find template $template_name"

  conda create --prefix $BUILDENV -c conda-forge -y python=3.6.9 pybind11 pytest spdlog=1.8.0 ninja cmake=3.18.2 protobuf=3.7.1 latexmk zip make conda-pack
  conda activate $BUILDENV
  pip install -r ${SRC}/requirements.txt

  # Build onnx for now (Waiting for https://github.com/conda-forge/staged-recipes/pull/13108 to be merged)
  mkdir $BUILD/onnx
  cd $BUILD/onnx
  curl -L https://github.com/onnx/onnx/archive/v1.7.0.tar.gz | tar zx --strip-components=1
  mkdir build && cd build
  cmake ../ -GNinja -DONNX_ML=0 -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  ninja install
  if [ "$CREATE_TEMPLATE" = true ]
  then
    conda pack -p $BUILDENV -o ${template_name}
  fi
else
  if [ "$CREATE_TEMPLATE" = true ]
  then
    echo "Found template $template_name: nothing to do"
    exit 0
  else
    echo "Found template $template_name: unpacking environment to $BUILDENV"
    mkdir $BUILDENV
    tar xf $template_name -C $BUILDENV
    source $BUILDENV/bin/activate
    conda-unpack
  fi
fi

echo "conda activate $BUILDENV" >> $BUILD/activate_buildenv.sh
