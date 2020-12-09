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
set(BUILD_DOCS OFF CACHE BOOL "Build PopTorch's documentation")

# Note: The next line is also parsed by scripts/utils/_utils.py
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

set(CMAKE_CXX_STANDARD 17)

set(INSTALL_PYDIR poptorch)

if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
  set(CMAKE_INSTALL_RPATH "@loader_path")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

add_subdirectory(poptorch_logging)
add_subdirectory(popart_compiler)
add_subdirectory(python)

# Examples and tests
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(docs/user_guide)

find_package(Torch ${TORCH_VERSION} EXACT REQUIRED)

add_custom_target(poptorch_wheel
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/setup.py sdist bdist_wheel
  DEPENDS poptorch_core
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        DESTINATION .)

if(BUILD_DOCS)
  set(PKG_EXTRA_FLAGS "--include-documentation")
endif()

# TODO(T27444): At the moment BUILD_DOCS is bypassed by the CI so instead
# check if the files exist with --include-documentation-if-exists
# TODO(T27444): move to Jenkins script: we don't need CMake for that
set(PKG_UPLOAD_DIR ../../pkg)
add_custom_target(package_and_move
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package_poptorch.py --output-dir ${PKG_UPLOAD_DIR} ${PKG_EXTRA_FLAGS} --include-documentation-if-exists --keep-output-dir
  DEPENDS poptorch_wheel
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/enable.sh.in
  ${PROJECT_BINARY_DIR}/enable.sh
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/enable.sh
        DESTINATION .)

if(${CMAKE_VERSION} VERSION_GREATER "3.15.0")
  # Building poptorch without installing it doesn't make sense: the python
  # module cannot be used so always install after a build.
  add_custom_target(install_poptorch ALL
      COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}
      DEPENDS poptorch_core
  )
endif()
