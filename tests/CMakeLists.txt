# Temporarily disabled due to a problem in popart:
# The exported `popart` target does not contain all its dependencies, like onnx.
# add_subdirectory(custom_ops)

find_package(popart REQUIRED)

find_library(POPLAR_LIB NAMES poplar HINTS ${POPLAR_DIR}/lib ${POPLAR_DIR}/lib64 REQUIRED)
get_filename_component(POPLAR_DIR ${POPLAR_LIB} DIRECTORY)
install(CODE
  "set(ENV{LD_LIBRARY_PATH} ${popart_LIB_DIR}:${POPLAR_DIR})
  execute_process( COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake --add_to_sys_path ${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} RESULT_VARIABLE RETVAL )
  if(\${RETVAL})
    message(FATAL_ERROR \"generate_test_file.py FAILED (See above)\")
  endif()")

