add_subdirectory(custom_ops)

install(CODE "execute_process( COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )")
