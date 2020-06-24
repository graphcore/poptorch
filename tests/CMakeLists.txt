function(add_poptorch_py_unit_test name)
  message(STATUS "Adding python test '${name}'")

  add_test(NAME "${name}_test"
           COMMAND python3 -m pytest -s ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()


add_poptorch_py_unit_test(batching)
add_poptorch_py_unit_test(bert_small_and_medium)
add_poptorch_py_unit_test(losses)
add_poptorch_py_unit_test(custom_loss)
add_poptorch_py_unit_test(norms)
add_poptorch_py_unit_test(optimizers)
add_poptorch_py_unit_test(resnets_inference)
add_poptorch_py_unit_test(shape_inference)
add_poptorch_py_unit_test(inputs)

#############
# C++ tests #
#############

find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# Ensure ABI matches that of PyTorch
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=${Torch_USE_CXX11_ABI}")

# Boost must be header only because the ABI must match PyTorch
find_package(Boost REQUIRED)
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

# We need to link against python as we are testing
# a python module
find_package(pybind11 REQUIRED)
find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} REQUIRED)

set(DEFAULT_TEST_VARIANTS "Cpu" CACHE STRING
    "The device variants to run tests on when no variants are set explicitly")

# Generate tests for these devices and variants. Cmake variable ENABLED_TEST_VARIANTS can
# be set with -DPOPTORCH_CMAKE_ARGS=-DENABLED_TEST_VARIANTS=Hw for Hw only tests, for example.
# -DPOPTORCH_CMAKE_ARGS="-DENABLED_TEST_VARIANTS=Cpu$<SEMICOLON>IpuModel$<SEMICOLON>Hw"
# Will enable all tests - note that you need to use '$<SEMICOLON>' not ';'
set(ENABLED_TEST_VARIANTS "Cpu;IpuModel" CACHE STRING "Default (non-hw) test variants")

# Remove any requested variants not present in ${ENABLED_TEST_VARIANTS}
# If no variants were requested filtered ${DEFAULT_TEST_VARIANTS} are returned.
function(sanitise_variants variant_name)
  set(variants ${${variant_name}})
  set(old_variants "${variants}")
  if (NOT old_variants)
    set(old_variants ${DEFAULT_TEST_VARIANTS})
  endif()
  set(new_variants "")
  # Only allow each variant if it is enabled:
  foreach(variant ${old_variants})
    if (${variant} IN_LIST ENABLED_TEST_VARIANTS)
      list(APPEND new_variants ${variant})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES new_variants)
  set(${variant_name} "${new_variants}" PARENT_SCOPE)
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "target" part storing it in the second argument.
function(extract_target variant target)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(GET vspec 0 HEAD)
  set(${target} ${HEAD} PARENT_SCOPE)
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts the "runconfig" part storing it in the second argument.
function(extract_runconfig variant run_config)
  string(REPLACE ":" ";" vspec ${${variant}})
  list(LENGTH vspec vlen)
  if (${vlen} EQUAL "1")
    set(${run_config} "default" PARENT_SCOPE)
  else()
    list(GET vspec 1 config)
    set(${run_config} ${config} PARENT_SCOPE)
  endif()
endfunction()

# A VARIANT is a colon separated pair "target:[runconfig]". This function
# extracts a list containing just the "target" parts. Where targets are listed
# with multiple runconfigs in the original list the duplicates are removed
# from the returned list.
function(extract_targets variants targets)
  set(extracted_targets "")
  foreach(variant ${${variants}})
    string(REPLACE ":" ";" vspec ${variant})
    list(GET vspec 0 HEAD)
    list(APPEND extracted_targets ${HEAD})
  endforeach()
  list(REMOVE_DUPLICATES extracted_targets)
  set(${targets} ${extracted_targets} PARENT_SCOPE)
endfunction()

function(add_test_executable name)
  add_executable(${name} ${ARGN} "${PROJECT_SOURCE_DIR}/shared/source/Logging.cpp")
  target_include_directories(${name}
      PRIVATE
      ${Boost_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${PROJECT_SOURCE_DIR}/poptorch/include"
      "${PROJECT_SOURCE_DIR}/shared/include"
      "${PROJECT_SOURCE_DIR}/popart_compiler/include"
      ${SPDLOG_INCLUDE_DIR}
      ${TORCH_INCLUDE_DIRS})

  target_link_libraries(${name}
    poptorch_core
    ${TORCH_LIBRARIES}
    popart_compiler
    ${PYTHON_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
  )
  # NB Boost must *not* be linked

  set_property(TARGET ${name}
              APPEND_STRING PROPERTY
              COMPILE_FLAGS "-DTEST_TARGET=TestDeviceType::${TEST_TARGET}")
endfunction()

# Optional arguments:
# VARIANTS
# Run with the specified VARIANTS (or all enabled VARIANTS if
# none are specified). The first parameter is the test name, the following are
# source files, and finally everything after VARIANTS are variant
# specifications.
#
# Mark the test as requiring two IPUs - the relevant fixture will be added
#
# Currently supported targets are "Cpu;IpuModel;Sim;Hw"
#
#
# LABELS
#
# Semicolon separated list of labels to be set for this test
#
#
# DUAL
# A variation specification must be of the form "target:[config]"
#
function(add_poptorch_cpp_unit_test name)
  set(oneValueArgs "VARIANTS;LABELS")
  cmake_parse_arguments(test_args "" "FILES" "${oneValueArgs}" "FILES;${ARGN}")
  set(FILES "${test_args_FILES}")

  sanitise_variants(test_args_VARIANTS)
  extract_targets(test_args_VARIANTS TARGETS)
  message(STATUS
    "Adding cpp test '${name}' with variants: ${test_args_VARIANTS}")

  foreach(TEST_TARGET ${TARGETS})
    set(executable_name "${TEST_TARGET}_${name}")
    add_test_executable(${executable_name} ${FILES})
  endforeach()

  foreach(VARIANT ${test_args_VARIANTS})
    extract_target(VARIANT TEST_TARGET)
    extract_runconfig(VARIANT TEST_CONFIG)

    set(test_name "${TEST_TARGET}_${TEST_CONFIG}_${name}")
    set(executable_name "${TEST_TARGET}_${name}")
    add_test(NAME "${test_name}"
      COMMAND ${executable_name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})

    set(test_env ${TEST_ENVIRONMENT})
    if (${TEST_CONFIG} STREQUAL "cpp")
      list(APPEND test_env "IPU_POPLAR_RT_GP_SUFFIX=_c")
    endif()

    set_tests_properties(${test_name} PROPERTIES
                         ENVIRONMENT "${test_env}")
    set_tests_properties(${test_name} PROPERTIES
                          LABELS "${test_args_LABELS}")
    if (${TEST_TARGET} STREQUAL "Hw")
      # Make sure tests that use physical IPUs only run if an appropriate
      # number were available according to the relevant test fixture:
      set_tests_properties(${test_name}
        PROPERTIES FIXTURES_REQUIRED SingleIpuIsAvailable)
    endif()
  endforeach()
endfunction()

# See T21802. C++ tests are disabled by default for now.
if (${POPTORCH_ENABLE_CPP_TESTS})
  add_poptorch_cpp_unit_test(exception_test exception_test.cpp)
endif()
