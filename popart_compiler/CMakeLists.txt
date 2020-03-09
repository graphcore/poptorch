cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(popart_compiler)

find_package(Popart REQUIRED)
find_package(Poplar REQUIRED)

add_library(popart_compiler SHARED "source/Compiler.cpp")

set_property(TARGET popart_compiler PROPERTY CXX_STANDARD 14)

target_link_libraries(popart_compiler PRIVATE ${POPART_LIB})

target_include_directories(popart_compiler PRIVATE
                                        "include"
                                        ${POPART_INCLUDE_DIR}
                                        ${POPLAR_INCLUDE_DIR})

install(TARGETS popart_compiler
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

