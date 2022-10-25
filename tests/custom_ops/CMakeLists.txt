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
  if(NOT poplar_FOUND)
    find_package(poplar REQUIRED)
  endif()
  if(NOT snap_FOUND)
    find_package(snap REQUIRED)
  endif()
else()
  # Check the package is not already in CMake's path
  find_package(poplar)
  if(NOT poplar_FOUND)
    message(FATAL_ERROR "You must provide a path to a Poplar install using -DPOPLAR_DIR=/path/to/popart/build/install")
  endif()
  find_package(snap)
  if(NOT snap_FOUND)
    message(FATAL_ERROR "You must provide a path to a Poplar install using -DPOPLAR_DIR=/path/to/popart/build/install")
  endif()
endif()
if( NOT EXISTS ${POPART_DIR} )
  # Check the package is not already in CMake's path
  find_package(popart COMPONENTS popart-only)
  if(NOT popart_FOUND)
    message(FATAL_ERROR "You must provide a path to a Popart build using -DPOPART_DIR=/path/to/popart/build")
  endif()
else()
  list(APPEND CMAKE_PREFIX_PATH ${POPART_DIR})
  if(NOT popart_FOUND)
    find_package(popart REQUIRED COMPONENTS popart-only)
  endif()
endif()

# All C++ code in this project will be compiled as C++14
set (CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(custom_cube_op SHARED "custom_cube_op.cpp")
target_link_libraries(custom_cube_op popart-only)

add_library(custom_leaky_relu_op SHARED "custom_leaky_relu_op.cpp")
target_link_libraries(custom_leaky_relu_op popart-only)

add_library(custom_add_scalar_op SHARED "custom_add_scalar_op.cpp")
target_link_libraries(custom_add_scalar_op popart-only)

add_library(custom_add_scalar_vec_op SHARED "custom_add_scalar_vec_op.cpp")
target_link_libraries(custom_add_scalar_vec_op popart-only)

add_library(custom_add_vec_scalar_mul_op SHARED
            "custom_add_vec_scalar_mul_op.cpp")
target_link_libraries(custom_add_vec_scalar_mul_op popart-only)

add_library(custom_reduce_op SHARED
            "custom_reduce_op.cpp")
target_link_libraries(custom_reduce_op popart-only)

add_library(custom_three_input_reduce_op SHARED
            "custom_three_input_reduce_op.cpp")
target_link_libraries(custom_three_input_reduce_op popart-only)

add_library(custom_many_attribute_op SHARED
            "custom_many_attribute_op.cpp")
target_link_libraries(custom_many_attribute_op popart-only)
