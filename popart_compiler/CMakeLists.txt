cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(popart_compiler)

find_package(popef REQUIRED)

add_library(popart_compiler_types INTERFACE)
target_include_directories(popart_compiler_types INTERFACE types/include)

add_library(popart_compiler SHARED
  "source/CodeletsCompilation.cpp"
  "source/Compiler.cpp"
  "source/CompilerImpl.cpp"
  "source/Utils.cpp"
  "source/SessionOptions.cpp"
  "source/custom_operations/Embedding.cpp"
  "source/custom_operations/FastGatherLastDim.cpp"
  "source/custom_operations/HostOp.cpp"
  "source/custom_operations/TorchSoftplus.cpp"
  "source/custom_operations/UpsampleBilinear2d.cpp"
)

file(GLOB_RECURSE popart_compiler_public_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp*" "${CMAKE_CURRENT_SOURCE_DIR}/types/include/*.hpp*")
set_target_properties(popart_compiler PROPERTIES
  CXX_STANDARD 14
  PUBLIC_HEADER "${popart_compiler_public_headers}")

target_link_libraries(popart_compiler PUBLIC
                                      popart_compiler_types
                                      PRIVATE
                                      popef
                                      popart-only
                                      poptorch_logging
                                      poptorch_exception_info
                                      poprithms)

target_include_directories(popart_compiler
                           PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                           PRIVATE
                           source/include)

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
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/popart_compiler
  )

foreach(SRC ${CUSTOM_CODELETS})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${SRC}
  DESTINATION ${INSTALL_PYDIR})
endforeach()
