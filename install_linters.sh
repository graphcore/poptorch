#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

set -e # Stop on error
DIR=$(realpath $(dirname $0))

# If some arguments were passed to the script assume this is coming from arc lint: print instructions and exit
if [[ $# -gt 0 ]]
then
  echo "Linters not installed: run ${DIR}/install_linters.sh to install the linters then try again"
  exit 1
fi

install_clang_format_apple() {
  CLANG_PLATFORM=darwin-apple
  BINARY=clang-format
  CLANG_VERSION=9.0.0
  FOLDER_NAME=clang+llvm-${CLANG_VERSION}-x86_64-${CLANG_PLATFORM}
  ARCHIVE_NAME=${FOLDER_NAME}.tar.xz
  URL=https://releases.llvm.org/${CLANG_VERSION}/${ARCHIVE_NAME}
  VERSIONED_BINARY="${DIR}/.linters/${BINARY}-${CLANG_VERSION}"

  if [ ! -f "${VERSIONED_BINARY}" ]; then
    TMP_DIR=`mktemp -d`
    pushd ${TMP_DIR}
    curl -O $URL
    tar xf ${ARCHIVE_NAME} ${FOLDER_NAME}/bin/clang-format
    mv ${FOLDER_NAME}/bin/clang-format ${VERSIONED_BINARY}
    chmod +x ${VERSIONED_BINARY}
    popd
    rm -rf ${TMP_DIR}
  fi
  ln -fs ${VERSIONED_BINARY} ${DIR}/.linters/clang-format
}

mkdir -p .linters

VE=${DIR}/.linters/venv

if [ ! -d ${VE} ]
then
  python3 -m venv ${VE}
fi

source ${VE}/bin/activate
pip install yapf==0.27.0
pip install cpplint==1.4.4

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  pip install clang-format==9.0.0
  ln -fs ${VE}/bin/clang-format ${DIR}/.linters/clang-format
elif [[ "$OSTYPE" == "darwin"* ]]; then
  install_clang_format_apple
else
  echo "ERROR: '${OSTYPE}' platform not supported"
  exit -1
fi

ln -fs ${VE}/bin/yapf ${DIR}/.linters/yapf
ln -fs ${VE}/bin/cpplint ${DIR}/.linters/cpplint

