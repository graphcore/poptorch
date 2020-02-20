cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(popart_compiler)


link_directories("${POPART_INSTALL}/lib")

add_library(popart_compiler SHARED "source/Compiler.cpp")

set_property(TARGET popart_compiler PROPERTY CXX_STANDARD 14)


target_link_libraries(popart_compiler PRIVATE popart)

target_include_directories(popart_compiler PRIVATE
                                        "include"
                                        "${POPART_INSTALL}/include")

