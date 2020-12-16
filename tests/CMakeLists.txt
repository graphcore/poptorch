add_subdirectory(custom_ops)

run_poptorch_install_command(
  "python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake --add-to-sys-path ${CMAKE_INSTALL_PREFIX}"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "generate_test_file.py")

