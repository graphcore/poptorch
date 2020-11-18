cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(poptorch)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Default to local install prefix" FORCE)
endif()

set(USE_PYTORCH_PACKAGE_HEADERS ON CACHE BOOL "Use the Torch headers distributed with the pytorch package.")
set(POPLAR_DIR CACHE PATH "Path to a Poplar install")
set(POPART_DIR CACHE PATH "Path to a Popart install")
set(SNAPSHOT "" CACHE STRING "Snapshot ID to use for the documentation")
set(SDK_DIR CACHE PATH "Path to an extracted SDK archive or to a Poplar & Popart install directory (Will populate POPLAR_DIR and POPART_DIR)")
set(USE_PROTOC_PROTOBUF ON CACHE BOOL "Use libprotobuf shipped with protoc instead of searching for it in the system path.")

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

if(USE_PROTOC_PROTOBUF)
  execute_process(COMMAND bash -c "dirname `which protoc`" OUTPUT_VARIABLE PROTOC_PATH)
  list(PREPEND CMAKE_PREFIX_PATH ${PROTOC_PATH}/..)
endif()

if(EXISTS ${SDK_DIR})
  execute_process(COMMAND find ${SDK_DIR} -maxdepth 1 -type d -name "popart*"
    OUTPUT_VARIABLE POPART_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND find ${SDK_DIR} -maxdepth 1 -type d -name "poplar-*" -o -name "poplar"
    OUTPUT_VARIABLE POPLAR_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT IS_DIRECTORY "${POPLAR_DIR}")
    message(FATAL_ERROR "Couldn't find a \"poplar\" or \"poplar-*\" folder in '${SDK_DIR}'")
  endif()
  if(NOT IS_DIRECTORY "${POPART_DIR}")
    message(FATAL_ERROR "Couldn't find a \"popart*\" folder in '${SDK_DIR}'")
  endif()
endif()

if(EXISTS ${POPLAR_DIR})
  list(APPEND CMAKE_PREFIX_PATH ${POPLAR_DIR})
  set(ENABLE_POPLAR_CMD "source ${POPLAR_DIR}/enable.sh")
else()
  # Check the package is not already in the path
  find_package(poplar)
  if(NOT poplar_FOUND)
    message(FATAL_ERROR "You must provide a path to a Poplar install using -DPOPLAR_DIR=/path/to/popart/build/install")
  endif()
endif()

if( EXISTS ${POPART_DIR} )
  list(APPEND CMAKE_PREFIX_PATH ${POPART_DIR})
  set(ENABLE_POPART_CMD "source ${POPART_DIR}/enable.sh")
else()
  find_package(popart)
  if(NOT popart_FOUND)
    message(FATAL_ERROR "You must provide a path to a Popart build using -DPOPART_DIR=/path/to/popart/build")
  endif()
endif()

file(WRITE ${CMAKE_BINARY_DIR}/tmp/test.sh "
  source ${CMAKE_BINARY_DIR}/enable.sh
  ctest --output-on-failure $@
  ")
file(COPY ${CMAKE_BINARY_DIR}/tmp/test.sh
  DESTINATION ${CMAKE_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

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
if("${SNAPSHOT}" STREQUAL "")
  execute_process(
    COMMAND git rev-parse --short=10 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SNAPSHOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if("${SNAPSHOT}" STREQUAL "")
  set(SHAPSHOT "0000000000")
endif()
message(STATUS "Snapshot is " ${SNAPSHOT})

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
add_subdirectory(docs/user_guide)

set(USER_GUIDE_PDF_NAME poptorch-user-guide-${VERSION}-${SNAPSHOT}.pdf)
set(USER_GUIDE_HTML_NAME poptorch-user-guide-html-${VERSION}-${SNAPSHOT}.zip)
set(PYTHON_API_PDF_NAME poptorch-python-api-${VERSION}-${SNAPSHOT}.pdf)
set(PYTHON_API_HTML_NAME poptorch-python-api-html-${VERSION}-${SNAPSHOT}.zip)

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

find_package(Torch ${TORCH_VERSION} EXACT REQUIRED)

configure_file(setup.py.in
  ${PROJECT_BINARY_DIR}/setup.py
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setup.py DESTINATION .)

execute_process(COMMAND python3 -c "import distutils.util;print(distutils.util.get_platform().replace('.','_').replace('-','_'), end='')"
  OUTPUT_VARIABLE PYTHON_PLATFORM)
set(PACKAGE_FILE_NAME poptorch-${VERSION}+${SNAPSHOT}+${PACKAGE_OS_TYPE}-py3-none-${PYTHON_PLATFORM}.whl)
set(PRODINFO_FILE_NAME poptorch-${VERSION}-${SNAPSHOT}.yml)

set(PKG_UPLOAD_DIR ../../pkg)

add_custom_target(poptorch_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 setup.py sdist bdist_wheel
  DEPENDS poptorch_core
)

set( COPY_PDF_IF_EXISTS
import os, shutil$<SEMICOLON>
yml = open('${PKG_UPLOAD_DIR}/${PRODINFO_FILE_NAME}','a')$<SEMICOLON>
os.path.exists('${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}')
and shutil.copyfile('${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}', '${PKG_UPLOAD_DIR}/${USER_GUIDE_PDF_NAME}')
and yml.write(' ' * 2 + '- name: PopTorch user guide \(PDF\)\\n'+' ' * 4 + 'file: ${USER_GUIDE_PDF_NAME}\\n' + ' ' * 4 + 'type: pdf\\n')
)

set( COPY_HTML_IF_EXISTS
import os, shutil$<SEMICOLON>
yml = open('${PKG_UPLOAD_DIR}/${PRODINFO_FILE_NAME}','a')$<SEMICOLON>
os.path.exists('${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_HTML_NAME}')
and shutil.copyfile('${CMAKE_CURRENT_BINARY_DIR}/${USER_GUIDE_HTML_NAME}', '${PKG_UPLOAD_DIR}/${USER_GUIDE_HTML_NAME}')
and yml.write(' ' * 2 + '- name: PopTorch user guide \(HTML\)\\n'+' ' * 4 + 'file: ${USER_GUIDE_HTML_NAME}\\n' + ' ' * 4 + 'type: html_zip\\n')
)

add_custom_target(package_and_move
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND bash -c "mkdir -p ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "cp ${CMAKE_CURRENT_BINARY_DIR}/${PRODINFO_FILE_NAME} ${PKG_UPLOAD_DIR}"
  COMMAND bash -c "cp dist/*.whl  ${PKG_UPLOAD_DIR}/${PACKAGE_FILE_NAME}"
  COMMAND python3 -c \"${COPY_PDF_IF_EXISTS}\"
  COMMAND python3 -c \"${COPY_HTML_IF_EXISTS}\"
  DEPENDS poptorch_wheel
)
#------ end of packaging

#------ product info file

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

if(${CMAKE_VERSION} VERSION_GREATER "3.15.0")
  # Building poptorch without installing it doesn't make sense: the python
  # module cannot be used so always install after a build.
  add_custom_target(install_poptorch ALL
      COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}
      DEPENDS poptorch_core
  )
endif()
