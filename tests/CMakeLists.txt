add_subdirectory(custom_ops)

# Copy tests to the build folder if requested.
if(COPY_TESTS)
  # NOTE: Collapsing the hierarchy like this may cause conflicts.
  file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
                       "${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/*.py"
                       "${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/**/*.py"
                       "${CMAKE_CURRENT_SOURCE_DIR}/gnn/*.py"
                       "${CMAKE_CURRENT_SOURCE_DIR}/gnn/**/*.py
                       ")
  install(FILES ${TEST_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
  set(TESTS_PATH "${CMAKE_CURRENT_BINARY_DIR}")
else()
  set(TESTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Generate tests.
run_poptorch_install_command(
  "python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py \
           ${TESTS_PATH} \
           ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake \
           --add-to-sys-path ${CMAKE_INSTALL_PREFIX} \
           --extra-pytest-args=\"${EXTRA_PYTEST_ARGS}\" "
  "${PROJECT_BINARY_DIR}"
  "generate_test_file.py")
