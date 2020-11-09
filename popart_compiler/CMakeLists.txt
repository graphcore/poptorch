cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(popart_compiler)

find_package(popart REQUIRED)
find_package(poplar REQUIRED)

add_definitions(-DONNX_NAMESPACE=onnx)

add_library(popart_compiler SHARED
  "source/Compiler.cpp"
  "source/PopartEnums.cpp"
  "source/Utils.cpp"
  )

set_property(TARGET popart_compiler PROPERTY CXX_STANDARD 14)

target_link_libraries(popart_compiler PRIVATE popart poptorch_logging poplar)

target_include_directories(popart_compiler PRIVATE "include")

install(TARGETS popart_compiler
  DESTINATION ${INSTALL_PYDIR})

