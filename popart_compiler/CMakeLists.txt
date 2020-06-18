cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(popart_compiler)

find_package(Boost REQUIRED)
find_package(Popart REQUIRED)
find_package(Poplar REQUIRED)

add_definitions(-DONNX_NAMESPACE=onnx)

add_library(popart_compiler SHARED
  "source/Compiler.cpp"
  )

set_property(TARGET popart_compiler PROPERTY CXX_STANDARD 14)

target_link_libraries(popart_compiler PRIVATE ${POPART_LIB} poptorch_logging)

target_include_directories(popart_compiler PRIVATE
                                        "include"
                                        "../poptorch_logging/include"
                                        ${SPDLOG_INCLUDE_DIR}
                                        ${POPART_INCLUDE_DIR}
                                        ${POPLAR_INCLUDE_DIR}
                                        ${POPLIBS_INCLUDE_DIR}
                                        ${Boost_INCLUDE_DIRS})

install(TARGETS popart_compiler
  DESTINATION ${INSTALL_PYDIR})

