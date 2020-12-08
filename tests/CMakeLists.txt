add_subdirectory(custom_ops)

find_package(popart REQUIRED)
find_package(poplar REQUIRED)

get_target_property(POPLAR_LIB poplar LOCATION)
get_filename_component(POPLAR_DIR ${POPLAR_LIB} DIRECTORY)
install(CODE
  "set(ENV{LD_LIBRARY_PATH} ${popart_LIB_DIR}:${POPLAR_DIR}:$ENV{LD_LIBRARY_PATH})
  execute_process( COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake --add-to-sys-path ${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} RESULT_VARIABLE RETVAL )
  if(\${RETVAL})
    message(FATAL_ERROR \"generate_test_file.py FAILED (See above)\")
  endif()")

