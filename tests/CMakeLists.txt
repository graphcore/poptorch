add_subdirectory(custom_ops)

# Copy tests to the build folder if requested.
if(COPY_TESTS)
  # NOTE: Collapsing the hierarchy like this may cause conflicts.
  file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
                       "${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/*.py"
                       "${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/**/*.py"
                       "${CMAKE_CURRENT_SOURCE_DIR}/mlir/lit*.cfg"
                       "${CMAKE_CURRENT_SOURCE_DIR}/mlir/*_test.mlir")
  install(FILES ${TEST_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
  set(TESTS_PATH "${CMAKE_CURRENT_BINARY_DIR}")
else()
  set(TESTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# If we're building with MLIR, generate MLIR pass tests by pointing to the
# necessary binaries.
set(POPTORCH_OPT_ARG "")
set(FILECHECK_ARG "")
set(POPTORCH_OPT_ARG "--poptorch-opt $<TARGET_FILE:poptorch-opt>")

find_program(FILECHECK_BIN FileCheck REQUIRED)
set(FILECHECK_ARG "--filecheck-bin ${FILECHECK_BIN}")

# Generate tests.
run_poptorch_install_command(
  "python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_test_file.py \
           ${TESTS_PATH} \
           ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake \
           ${POPTORCH_OPT_ARG} ${FILECHECK_ARG} \
           --add-to-sys-path ${CMAKE_INSTALL_PREFIX} \
           --extra-pytest-args=\"${EXTRA_PYTEST_ARGS}\" "
  "${PROJECT_BINARY_DIR}"
  "generate_test_file.py")
