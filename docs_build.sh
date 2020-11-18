#!/bin/bash -eu
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# realpath doesn't exist on osx
realpath() {
  python3 -c "import os.path; print(os.path.realpath('$1'))"
}
SRC=$(realpath $(dirname $0))

POPTORCH_BUILD_DIR=$(pwd)
DOCS_SOURCE_DIR="${SRC}/docs/user_guide"
DOCS_BUILD_DIR="${POPTORCH_BUILD_DIR}/docs"
SPHINX_CONF_DIR="${SRC}/docs/common"
DOC="poptorch-user-guide"
if ! python3 -c "import poptorch"
then
  echo "poptorch must be in your PYTHONPATH to generate the documentation: did you enable your build environment?"
  exit -1
fi
if ! PACKAGE_INFO_FILES=($(ls ${POPTORCH_BUILD_DIR}/poptorch-*.yml)) 
then
  echo "ERROR: You need to call $0 from the folder containing the package YAML info file"
  exit -1
fi
if (( ${#PACKAGE_INFO_FILES[@]} > 1 )); then
  echo "ERROR: More than one package info file: ${PACKAGE_INFO_FILES[*]}"
  exit -1
fi
PACKAGE_INFO_FILE=${PACKAGE_INFO_FILES[0]}
if [[ $PACKAGE_INFO_FILE =~ ^.*poptorch-(.*)-(.*).yml$ ]]
then
  VERSION=${BASH_REMATCH[1]}
  SNAPSHOT=${BASH_REMATCH[2]}
else
  echo "ERROR: Failed to extract version and snapshot from filename $PACKAGE_INFO_FILE"
  exit -1
fi
USER_GUIDE_PDF_NAME="${DOC}-${VERSION}-${SNAPSHOT}.pdf"
USER_GUIDE_HTML_NAME="${DOC}-html-${VERSION}-${SNAPSHOT}.zip"
PACKAGE_INFO_FILE="${POPTORCH_BUILD_DIR}/poptorch-${VERSION}-${SNAPSHOT}.yml"

rm -rf ${DOCS_BUILD_DIR}
mkdir -p "${DOCS_BUILD_DIR}/html/${DOC}"

TITLE=$(grep -m 1 . "${DOCS_SOURCE_DIR}/index.rst")

# Build HTML
set -e # Stop on error
# -a  write all files (default: only write new and changed files)
# -E don't use a saved environment, always read all files
# -n nit-picky mode, warn about all missing references
# -W turn warnings into errors
common_flags="-a -E -n -W --keep-going -j auto"

sphinx-build $common_flags -b html -c ${SPHINX_CONF_DIR} -D "project=${TITLE}" -D "html_title=${TITLE}" -D "version=v${VERSION}" "${DOCS_SOURCE_DIR}" "${DOCS_BUILD_DIR}/html/${DOC}"
# And a zip file of the html
( cd "${DOCS_BUILD_DIR}/html/${DOC}" && zip -q -r html.zip ./*.html ./*.js _images _static )
cp "${DOCS_BUILD_DIR}/html/${DOC}/html.zip" "${POPTORCH_BUILD_DIR}/${USER_GUIDE_HTML_NAME}"

# Build PDF
DOC_TITLE="${TITLE}" sphinx-build $common_flags -b latex -c ${SPHINX_CONF_DIR} -D "project=${DOC}" -D "version=v${VERSION}" -D "release=v${VERSION}" "${DOCS_SOURCE_DIR}" "${DOCS_BUILD_DIR}/latex/${DOC}"
( cd "${DOCS_BUILD_DIR}/latex/${DOC}" && make LATEXMKOPTS="-silent" )
cp "${DOCS_BUILD_DIR}/latex/${DOC}/doc.pdf" "${POPTORCH_BUILD_DIR}/${USER_GUIDE_PDF_NAME}"

