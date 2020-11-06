set(SDK_DIR CACHE PATH "Path to an extracted SDK archive or to a Poplar & Popart install directory")
set(ENABLE_WERROR TRUE CACHE BOOL "Stop compilation on warning (-Werror)")
set(POPLAR_INSTALL_DIR CACHE PATH "Path to a Poplar install")

#TODO(T27444): Remove popart from cbt.json
if(NOT EXISTS ${SDK_DIR})
  #TODO(T27444): When switching to Poptorch view, enable the message below and remove the "if"
  #message(FATAL_ERROR "You need to provide a Poplar or an SDK build: try -DSDK_DIR=/path/to/poplar_view/build/install")
  if(NOT EXISTS ${POPLAR_INSTALL_DIR})
    message(FATAL_ERROR "You need to provide either a Poplar or a SDK install:\
    try -DSDK_DIR=/path/to/poplar_view/build/install if you are building PopTorch\
    on its own or -DPOPLAR_INSTALL_DIR=/path/to/poplar_view/build/install if you\
    are building the POPONNX view.")
  endif()
  set(POPART_DIR ${CMAKE_BINARY_DIR}/install/ CACHE PATH "Path to a Popart install")
else()
  execute_process(COMMAND find ${SDK_DIR} -maxdepth 1 -type d -name "popart*"
    OUTPUT_VARIABLE POPART_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND find ${SDK_DIR} -maxdepth 1 -type d -name "poplar-*" -o -name "poplar"
    OUTPUT_VARIABLE POPLAR_INSTALL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT IS_DIRECTORY "${POPLAR_INSTALL_DIR}")
    message(FATAL_ERROR "Couldn't find a \"poplar\" or \"poplar-*\" folder in '${SDK_DIR}'")
  endif()
  if(NOT IS_DIRECTORY "${POPART_DIR}")
    message(FATAL_ERROR "Couldn't find a \"popart*\" folder in '${SDK_DIR}'")
  endif()
endif()

execute_process(
  COMMAND git rev-parse --short=10 HEAD
  WORKING_DIRECTORY ${CBT_DIR}/..
  OUTPUT_VARIABLE SNAPSHOT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND POPTORCH_CMAKE_ARGS -DPOPLAR_DIR=${POPLAR_INSTALL_DIR})
list(APPEND POPTORCH_CMAKE_ARGS -DPOPART_DIR=${POPART_DIR})
list(APPEND POPTORCH_CMAKE_ARGS -DENABLE_WERROR=${ENABLE_WERROR})
list(APPEND POPTORCH_CMAKE_ARGS -DSNAPSHOT=${SNAPSHOT})
list(APPEND POPTORCH_CMAKE_ARGS -DUSE_ENV_PROTOBUF=OFF)
list(APPEND POPTORCH_CMAKE_ARGS -DProtobuf_ROOT=${CMAKE_BINARY_DIR}/install/protobuf)

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

add_custom_target(poptorch_wheel COMMAND
  ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/build/poptorch --target poptorch_wheel DEPENDS poptorch)

add_custom_target(package_poptorch
  COMMAND bash -c '${CBT_DIR}/../poptorch/docs_build.sh'
  COMMAND ${CMAKE_COMMAND} --build . --target package_and_move
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/build/poptorch
  DEPENDS poptorch)
