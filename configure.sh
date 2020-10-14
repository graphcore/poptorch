#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
DIR=$(realpath $(dirname $0))
if [ ${DIR} = $(realpath $(pwd)) ]
then
  echo "Warning: This script needs to be called from a build directory"
  echo "Trying to use ../build"
  mkdir ../build
  cd ../build
fi
python3 -m venv venv
source venv/bin/activate
pip install -r ${DIR}/requirements.txt
