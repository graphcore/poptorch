cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
include(GNUInstallDirs)
project(poptorch)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 17)

set(INSTALL_PYDIR python)
add_subdirectory(popart_compiler)
add_subdirectory(python)
