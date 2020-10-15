cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(poptorch)

set(USE_PYTORCH_PACKAGE_HEADERS ON CACHE BOOL "Use the Torch headers distributed with the pytorch package.")
set(USE_ONNX_PACKAGE_HEADERS ON CACHE BOOL "Use the ONNX headers distributed with the ONNX pip package.")
set(POPART_DIR CACHE PATH "Path to a Popart install")

set(TORCH_VERSION 1.6.0)

set(CMAKE_CONFIGURATION_TYPES "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
# Enable IN_LIST operator
cmake_policy(SET CMP0057 NEW)
if(NOT CMAKE_BUILD_TYPE)
  list(GET CMAKE_CONFIGURATION_TYPES 0 CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
endif()
if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
  message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of ${CMAKE_CONFIGURATION_TYPES}")
endif()

if(USE_PYTORCH_PACKAGE_HEADERS)
  execute_process(COMMAND python3 -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"
    OUTPUT_VARIABLE TORCH_PATH)
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
  if(NOT TORCH_PATH)
    message(FATAL_ERROR "python3 -c \"import torch\" failed: check that your virtual environment is active and torch is installed")
  endif()
endif()

if(NOT DEFINED ONNX_DIR AND USE_ONNX_PACKAGE_HEADERS)
  execute_process(COMMAND python3 -c "import onnx; from pathlib import Path; print(Path(onnx.__file__).parent.parent, end='')"
    OUTPUT_VARIABLE ONNX_PATH)
  if(NOT ONNX_PATH)
    message(FATAL_ERROR "python3 -c \"import onnx\" failed: check that your virtual environment is active and onnx is installed")
  endif()
endif()

if( NOT EXISTS ${POPART_DIR} )
  #TODO(T27444): Re-enable once PopTorch has its own view.
  #message(FATAL_ERROR "You must provide a path to a Popart build using -DPOPART_DIR=/path/to/popart/build")
else()
  list(APPEND CMAKE_PREFIX_PATH ${POPART_DIR})
endif()


enable_testing()

add_compile_options(
  -Wall
  -pedantic
  -Wextra
  -Wdisabled-optimization
  -Wshadow
  -Wswitch
  -Wformat=2
  -Wimplicit-fallthrough
  -Winit-self
  -Wcomment
  -Wsequence-point
  -Wundef
  -Wuninitialized)

if(${ENABLE_WERROR})
  add_compile_options(-Werror)
endif()
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

set(DOCS_VERSION "${VERSION}")
if (DEFINED BUILD_NUMBER)
  set(VERSION "${VERSION}+${BUILD_NUMBER}")
endif()
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

#------ git information
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

execute_process(COMMAND
  git rev-parse --short=10 HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
#------ end of git information

set(INSTALL_PYDIR poptorch)

if(${OS_TYPE} STREQUAL osx)
  set(CMAKE_INSTALL_RPATH "@loader_path")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
elseif(${OS_TYPE} STREQUAL linux)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

add_subdirectory(poptorch_logging)
add_subdirectory(popart_compiler)
add_subdirectory(python)

# Examples and tests
add_subdirectory(tests)
add_subdirectory(examples)

set(USER_GUIDE_PDF_NAME poptorch-user-guide-${VERSION}-${SNAPSHOT}.pdf)
set(USER_GUIDE_HTML_NAME poptorch-user-guide-html-${VERSION}-${SNAPSHOT}.zip)
set(PYTHON_API_PDF_NAME poptorch-python-api-${VERSION}-${SNAPSHOT}.pdf)
set(PYTHON_API_HTML_NAME poptorch-python-api-html-${VERSION}-${SNAPSHOT}.zip)

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

find_package(Torch ${TORCH_VERSION} REQUIRED)

configure_file(setup.py.in
  ${PROJECT_BINARY_DIR}/setup.py
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setup.py DESTINATION .)

set(PACKAGE_FILE_NAME poptorch-${VERSION}+${SNAPSHOT}+${PACKAGE_OS_TYPE}-py3-none-linux_x86_64.whl)
set(PRODINFO_FILE_NAME poptorch-${VERSION}-${SNAPSHOT}.yml)

set(PKG_UPLOAD_DIR ../../pkg)

add_custom_target(poptorch_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 setup.py sdist bdist_wheel
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
)

add_custom_target(package_and_move
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv ${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "mv dist/*.whl  ${PKG_UPLOAD_DIR}/${PACKAGE_FILE_NAME}"
  COMMAND bash -c '[ -f \"${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}\" ] && mv \"${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}\" \"${PKG_UPLOAD_DIR}\"'
  COMMAND bash -c '[ -f \"${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_HTML_NAME}\" ] && mv \"${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_HTML_NAME}\" \"${PKG_UPLOAD_DIR}\"'
  DEPENDS poptorch_wheel
)
#------ end of packaging

#------ product info file
if(BUILD_DOCS)
  string(APPEND PKG_ELEMENTS "  - name: PopTorch user guide (PDF)\n"
                             "    file: ${USER_GUIDE_PDF_NAME}\n"
                             "    type: pdf\n")

  string(APPEND PKG_ELEMENTS "  - name: PopTorch user guide (HTML)\n"
                             "    file: ${USER_GUIDE_HTML_NAME}\n"
                             "    type: html_zip\n")
endif()

execute_process(
  COMMAND git config --get remote.origin.url
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  OUTPUT_VARIABLE VIEW_REMOTE_URL
  OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(poptorch.prodinfo.yml
  ${CMAKE_CURRENT_BINARY_DIR}/${PRODINFO_FILE_NAME}
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PRODINFO_FILE_NAME}
        DESTINATION .)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        DESTINATION .)

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

