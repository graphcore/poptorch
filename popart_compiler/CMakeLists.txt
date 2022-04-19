cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(popart_compiler)

find_package(popef REQUIRED)

add_library(popart_compiler SHARED
  "source/CodeletsCompilation.cpp"
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

target_link_libraries(popart_compiler PRIVATE popef popart-only poptorch_logging)

target_include_directories(popart_compiler PUBLIC include PRIVATE source/include)

# Copy custom codelet sources so that we can install and later pre-compile them
# on-demand, configure_file keeps track of changes and always copies on new
# version. Custom codelets are also copied into the python package during wheel
# creation.
set(CUSTOM_CODELETS
  "UpsampleBilinear2dCodelets.inc.cpp"
  "FastGatherLastDimFwdCodelets.inc.cpp"
  "FastGatherLastDimBwdCodelets.inc.cpp"
)

foreach(SRC ${CUSTOM_CODELETS})
  configure_file(source/custom_operations/${SRC} ${SRC} COPYONLY)
endforeach()

install(TARGETS popart_compiler
  DESTINATION ${INSTALL_PYDIR})

foreach(SRC ${CUSTOM_CODELETS})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${SRC}
    DESTINATION ${INSTALL_PYDIR})
endforeach()
