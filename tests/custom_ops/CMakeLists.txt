# This compiles a shared object file for the cube operator.
# Run make in the custom_ops folder to build. 
cmake_minimum_required(VERSION 3.4)
project(custom_cube_op)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ONNX_DIR ${ONNX_PATH} CACHE PATH "Path to the onnx sources or build")

# All C++ code in this project will be compiled as C++14
set (CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_definitions(-DONNX_NAMESPACE=onnx)

find_package(popart REQUIRED)
find_package(poplar REQUIRED)

find_path(ONNX_SCHEMA_INCLUDE_DIR
  NAMES onnx/defs/schema.h
  HINTS ${ONNX_DIR}
  DOC "directory containing the header with functions \
  for checking opset:version mapping, schema.h")

if(NOT ONNX_SCHEMA_INCLUDE_DIR)
message(FATAL_ERROR "Could not find 'onnx/defs/schema.h' in '${ONNX_DIR}', Try setting \
ONNX_DIR if not already done, something like \
-DONNX_DIR=/path/to/onnx/ ${ONNX_SCHEMA_INCLUDE_DIR}")
endif()

add_library(custom_cube_op SHARED "custom_cube_op.cpp")

target_include_directories(custom_cube_op SYSTEM PRIVATE ${ONNX_SCHEMA_INCLUDE_DIR})

target_link_libraries(custom_cube_op popart poplar)
