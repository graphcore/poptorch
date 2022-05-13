add_subdirectory(custom_ops)

if(COPY_TESTS)
  file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/popart/*.py")
  install(FILES ${TEST_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
  set(TESTS_PATH "${CMAKE_CURRENT_BINARY_DIR}")
else()
  set(TESTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

 # DEPRECATED: will be removed in a separate commit
if(ONLY_RUN_HW_TESTS)
  set(EXTRA_PYTEST_ARGS "--hw-tests-only")
else()
  set(EXTRA_PYTEST_ARGS "")
endif()

run_poptorch_install_command(
  "python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py ${TESTS_PATH} ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake --add-to-sys-path ${CMAKE_INSTALL_PREFIX} ${EXTRA_PYTEST_ARGS}"
  "${PROJECT_BINARY_DIR}"
  "generate_test_file.py")
