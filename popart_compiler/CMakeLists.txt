cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(popart_compiler)


add_library(popart_compiler SHARED
  "source/Compiler.cpp"
  "source/CompilerImpl.cpp"
  "source/PopartEnums.cpp"
  "source/Utils.cpp"
  "source/SessionOptions.cpp"
  "source/custom_operations/Embedding.cpp"
  "source/custom_operations/FastGatherLastDim.cpp"
  "source/custom_operations/HostOp.cpp"
  "source/custom_operations/TorchSoftplus.cpp"
  "source/custom_operations/UpsampleBilinear2d.cpp"
)

set_property(TARGET popart_compiler PROPERTY CXX_STANDARD 14)

target_link_libraries(popart_compiler PRIVATE popart-only poptorch_logging)

target_include_directories(popart_compiler PUBLIC include PRIVATE source/include)

install(TARGETS popart_compiler
  DESTINATION ${INSTALL_PYDIR})

