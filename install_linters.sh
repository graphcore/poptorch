#!/bin/bash

set -e # Stop on error
DIR=$(realpath $(dirname $0))

# If some arguments were passed to the script assume this is coming from arc lint: print instructions and exit
if [[ $# -gt 0 ]]
then
  echo "Linters not installed: run ${DIR}/install_linters.sh to install the linters then try again"
  exit 1
fi
mkdir -p .linters

VE=${DIR}/.linters/venv

if [ ! -d ${VE} ]
then
  python3 -m venv ${VE}
fi

source ${VE}/bin/activate
pip install yapf==0.27.0
pip install cpplint==1.4.4
pip install clang-format==9.0.0

ln -fs ${VE}/bin/yapf ${DIR}/.linters/yapf
ln -fs ${VE}/bin/cpplint ${DIR}/.linters/cpplint
ln -fs ${VE}/bin/clang-format ${DIR}/.linters/clang-format

