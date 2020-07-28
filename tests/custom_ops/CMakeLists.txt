# This compiles a shared object file for the cube operator.
# Run make in the custom_ops folder to build. 
cmake_minimum_required(VERSION 3.4)
project(custom_cube_op)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# All C++ code in this project will be compiled as C++14
set (CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


#Where to find Findxxxx.cmake files
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../popart/popart/cmake/Modules ${PROJECT_SOURCE_DIR}/../../../popart/popart/cmake)

add_definitions(-DONNX_NAMESPACE=onnx)

find_package(ONNX REQUIRED)
find_package(PROTOBUF REQUIRED)
find_package(poplar REQUIRED)
find_package(Popart REQUIRED)

link_directories("${POPLAR_INSTALL_DIR}/lib")

add_library(custom_cube_op SHARED "custom_cube_op.cpp")

target_include_directories(custom_cube_op SYSTEM PRIVATE
                                        ${ONNX_SCHEMA_INCLUDE_DIR}
                                        ${POPART_INCLUDE_DIR}
                                        ${POPLAR_INCLUDE_DIR}
                                        ${POPLIBS_INCLUDE_DIR}
                                        ${PROTOBUF_INCLUDE_DIR})

target_link_libraries(custom_cube_op ${POPART_LIB} poplar popops poplin popnn poprand)
