cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(poptorch_err)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(poptorch_exception_info INTERFACE)

target_include_directories(poptorch_exception_info
                           INTERFACE
                           exception_info)

add_library(poptorch_err STATIC
  "source/ExceptionHandling.cpp")

target_include_directories(poptorch_err SYSTEM PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                                        )
file(GLOB_RECURSE poptorch_err_public_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp*" "exception_info/*.hpp*")

set_target_properties(poptorch_err PROPERTIES
  PUBLIC_HEADER "${poptorch_err_public_headers}")
target_link_libraries(poptorch_err
                      PUBLIC
                      torch
                      poptorch_exception_info
                      PRIVATE
                      popart_compiler
                      poptorch_logging)

install(TARGETS poptorch_err
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/poptorch_err
  )
