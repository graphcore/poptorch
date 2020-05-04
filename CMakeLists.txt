cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
include(GNUInstallDirs)
project(poptorch)
set(VERSION "0.0.0")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 17)

#with cmake >= 3.12 this should be add_compile_definitions(ONNX_NAMESPACE=onnx)
add_definitions(-DONNX_NAMESPACE=onnx)

set(INSTALL_PYDIR python)
add_subdirectory(popart_compiler)
add_subdirectory(python)

add_subdirectory(docs)
