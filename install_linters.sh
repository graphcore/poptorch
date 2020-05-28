#!/bin/bash

set -e # Stop on error
DIR=$(realpath $(dirname $0))

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  CLANG_PLATFORM=linux-gnu-ubuntu-18.04
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # Mac OSX
  CLANG_PLATFORM=darwin-apple
else
  echo "ERROR: '${OSTYPE}' platform not supported"
  exit -1
fi

install_clang_format() {
  BINARY=clang-format
  CLANG_VERSION=9.0.0
  FOLDER_NAME=clang+llvm-${CLANG_VERSION}-x86_64-${CLANG_PLATFORM}
  ARCHIVE_NAME=${FOLDER_NAME}.tar.xz
  URL=http://releases.llvm.org/${CLANG_VERSION}/${ARCHIVE_NAME}
  VERSIONED_BINARY="${DIR}/.linters/${BINARY}-${CLANG_VERSION}"
  SYMLINK="${DIR}/.linters/${BINARY}"

  if [ ! -f "${VERSIONED_BINARY}" ]; then
    TMP_DIR=`mktemp -d`
    pushd ${TMP_DIR}
    wget $URL --progress=dot:mega -o /dev/stdout
    tar xf ${ARCHIVE_NAME} ${FOLDER_NAME}/bin/clang-format
    mv ${FOLDER_NAME}/bin/clang-format ${VERSIONED_BINARY}
    chmod +x ${VERSIONED_BINARY}
    popd
    rm -rf ${TMP_DIR}
  fi
  ln -fs ${VERSIONED_BINARY} ${SYMLINK}
}

mkdir -p .linters

install_clang_format

VE=${DIR}/.linters/venv

if [ ! -d ${VE} ]
then
  python3 -m venv ${VE}
fi

source ${VE}/bin/activate
pip install yapf==0.27.0
pip install cpplint==1.4.4
ln -fs ${VE}/bin/yapf ${DIR}/.linters/yapf
ln -fs ${VE}/bin/cpplint ${DIR}/.linters/cpplint

