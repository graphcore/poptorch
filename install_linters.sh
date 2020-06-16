#!/bin/bash

set -e # Stop on error
DIR=$(realpath $(dirname $0))

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

