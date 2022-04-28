cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(poptorch_err)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Packages provided by Poplar
find_package(pybind11 REQUIRED)

add_library(poptorch_exception_info INTERFACE)

target_include_directories(poptorch_exception_info
                           INTERFACE
                           exception_info)

add_library(poptorch_err STATIC
  "source/ExceptionHandling.cpp")

target_include_directories(poptorch_err SYSTEM PUBLIC
                                        include
                                        )
target_link_libraries(poptorch_err
                      PUBLIC
                      torch
                      poptorch_exception_info
                      PRIVATE
                      pybind11::module
                      popart_compiler
                      poptorch_logging)
