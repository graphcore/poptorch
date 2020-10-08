add_subdirectory(custom_ops)

set(EXCLUDE_LONG_TESTS OFF CACHE BOOL "Do not run the tests taking over 4min")

install(CODE "execute_process( COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake --exclude-long-tests=${EXCLUDE_LONG_TESTS} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )")
