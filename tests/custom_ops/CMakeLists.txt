# This compiles a shared object file for the cube operator.
# Run make in the custom_ops folder to build. 
cmake_minimum_required(VERSION 3.4)
project(custom_cube_op)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(POPART_DIR CACHE PATH "Path to a Popart install")
set(POPLAR_DIR CACHE PATH "Path to a Poplar install")

if( NOT ${POPLAR_DIR} STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH ${POPLAR_DIR})
else()
  # Check the package is not already in CMake's path
  find_package(poplar)
  if(NOT poplar_FOUND)
    message(FATAL_ERROR "You must provide a path to a Poplar install using -DPOPLAR_DIR=/path/to/popart/build/install")
  endif()
endif()
if( NOT EXISTS ${POPART_DIR} )
  # Check the package is not already in CMake's path
  find_package(popart)
  if(NOT popart_FOUND)
    message(FATAL_ERROR "You must provide a path to a Popart build using -DPOPART_DIR=/path/to/popart/build")
  endif()
else()
  list(APPEND CMAKE_PREFIX_PATH ${POPART_DIR})
endif()

# All C++ code in this project will be compiled as C++14
set (CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

cmake_policy(SET CMP0074 NEW) # Check Protobuf_ROOT
find_package(popart REQUIRED)

add_library(custom_cube_op SHARED "custom_cube_op.cpp")

# For some reason protobuf::libprotobuf doesn't work. Protobuf comes from popart.
target_include_directories(custom_cube_op SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
target_link_libraries(custom_cube_op popart)
