cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
include(GNUInstallDirs)
project(poptorch)

#------ release version numnber
file(READ ${PROJECT_SOURCE_DIR}/version.json VERSION_JSON)
string(REGEX REPLACE
       ".*major[^:]*: \"([^\"]*)\".*" "\\1"
       CPACK_PACKAGE_VERSION_MAJOR
       ${VERSION_JSON})
string(REGEX REPLACE
       ".*minor[^:]*: \"([^\"]*)\".*" "\\1"
       CPACK_PACKAGE_VERSION_MINOR
       ${VERSION_JSON})
string(REGEX REPLACE
       ".*point[^:]*: \"([^\"]*)\".*" "\\1"
       CPACK_PACKAGE_VERSION_PATCH
       ${VERSION_JSON})

set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR})
set(VERSION ${VERSION}.${CPACK_PACKAGE_VERSION_MINOR})
set(VERSION ${VERSION}.${CPACK_PACKAGE_VERSION_PATCH})
#------ end of release version numnber

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 17)

#------ OS version info
if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
  set(OS_TYPE osx)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
  set(OS_TYPE linux)
else()
  set(OS_TYPE unknown)
endif()

#with cmake >= 3.12 this should be add_compile_definitions(ONNX_NAMESPACE=onnx)
add_definitions(-DONNX_NAMESPACE=onnx)

#------ cpack information
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  OUTPUT_VARIABLE VIEW_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if("${VIEW_HASH}" STREQUAL "")
  set(VIEW_SHORT_HASH "0000000000")
else()
  string(SUBSTRING ${VIEW_HASH} 0 10 VIEW_SHORT_HASH)
endif()
message(STATUS "View hash is " ${VIEW_SHORT_HASH})

set(SNAPSHOT ${VIEW_SHORT_HASH})

set(CPACK_PACKAGE_VENDOR graphcore)
set(CPACK_GENERATOR TGZ)

execute_process(COMMAND
  git rev-parse --short=10 HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
#------ end of cpack information

set(INSTALL_PYDIR python)
add_subdirectory(popart_compiler)
add_subdirectory(python)

# Examples and tests
#add_subdirectory(tests)
#add_subdirectory(examples)

add_subdirectory(docs)

#------ packaging
if(${OS_TYPE} STREQUAL osx)
  set(PACKAGE_OS_TYPE osx)
elseif(${OS_TYPE} STREQUAL linux)
  # For now we assume we are packaging for ubuntu 18.04, this
  # will have to change at a later date to a be a parameter of the build
  set(PACKAGE_OS_TYPE ubuntu_18_04)
else()
  set(PACKAGE_OS_TYPE unknown)
endif()

# SWDB_PACKAGE_NAME is used to set the name of the main poplar package,
# e.g. ubuntu_16_04_installer
if (DEFINED SWDB_PACKAGE_NAME)
  set(PACKAGE_OS_TYPE ${SWDB_PACKAGE_NAME})
  string(REPLACE "_installer" "" PACKAGE_OS_TYPE ${PACKAGE_OS_TYPE})
  message(STATUS "Overriding package type '${PACKAGE_OS_TYPE}'")
endif()

set(PACKAGE_FILE_NAME poptorch-${PACKAGE_OS_TYPE}-${VERSION}-${SNAPSHOT})
set(CPACK_PACKAGE_FILE_NAME ${PACKAGE_FILE_NAME})

set(USER_GUIDE_PDF_NAME poptorch-user-guide-${VERSION}-${SNAPSHOT}.pdf)
set(USER_GUIDE_HTML_NAME poptorch-user-guide-html-${VERSION}-${SNAPSHOT}.zip)
set(PYTHON_API_PDF_NAME poptorch-python-api-${VERSION}-${SNAPSHOT}.pdf)
set(PYTHON_API_HTML_NAME poptorch-python-api-html-${VERSION}-${SNAPSHOT}.zip)
set(PRODINFO_FILE_NAME poptorch-${VERSION}-${SNAPSHOT}.yml)

include(CPack)

set(PKG_UPLOAD_DIR ../../pkg)

if(BUILD_DOCS)
add_custom_target(package_and_move
  COMMAND ${CMAKE_CPACK_COMMAND} --config ${CMAKE_BINARY_DIR}/CPackConfig.cmake
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
#  COMMAND bash -c "mv ${USER_GUIDE_PDF_NAME} ${PKG_UPLOAD_DIR}"
#  COMMAND bash -c "mv ${USER_GUIDE_HTML_NAME} ${PKG_UPLOAD_DIR}"
#  COMMAND bash -c "mv ${PYTHON_API_PDF_NAME} ${PKG_UPLOAD_DIR}"
#  COMMAND bash -c "mv ${PYTHON_API_HTML_NAME} ${PKG_UPLOAD_DIR}"
)
else()
add_custom_target(package_and_move
  COMMAND ${CMAKE_CPACK_COMMAND} --config ${CMAKE_BINARY_DIR}/CPackConfig.cmake
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PACKAGE_FILE_NAME}.tar.gz ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
)
endif()
#------ end of packaging

#------ product info file
if(BUILD_DOCS)
#  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime user guide (PDF)\n"
#                             "    file: ${USER_GUIDE_PDF_NAME}\n"
#                             "    type: pdf\n")
#
#  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime user guide (HTML)\n"
#                             "    file: ${USER_GUIDE_HTML_NAME}\n"
#                             "    type: html_zip\n")
#
#  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime Python API (PDF)\n"
#                             "    file: ${PYTHON_API_PDF_NAME}\n"
#                             "    type: pdf\n")
#
#  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime Python API (HTML)\n"
#                             "    file: ${PYTHON_API_HTML_NAME}\n"
#                             "    type: html_zip\n")
#
#  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime C++ API (PDF)\n"
#                             "    file: ${CPP_API_PDF_NAME}\n"
#                             "    type: pdf\n")
#
#  string(APPEND PKG_ELEMENTS "  - name: Poplar advanced runtime C++ API (HTML)\n"
#                             "    file: ${CPP_API_HTML_NAME}\n"
#                             "    type: html_zip\n")
endif()

execute_process(
  COMMAND git config --get remote.origin.url
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  OUTPUT_VARIABLE VIEW_REMOTE_URL
  OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(poptorch.prodinfo.yml
  ${CMAKE_CURRENT_BINARY_DIR}/${PRODINFO_FILE_NAME}
  @ONLY)
#------ end of product info file

#------ enable script
if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
  set(LD_PATH_VARNAME "DYLD_LIBRARY_PATH")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
  set(LD_PATH_VARNAME "LD_LIBRARY_PATH")
else()
  message(FATAL_ERROR "Unknown system name")
endif()

configure_file(enable.sh.in
  ${PROJECT_BINARY_DIR}/enable.sh
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/enable.sh
        DESTINATION .)
#------ end of enable script
