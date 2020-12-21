# Enable IN_LIST operator
cmake_policy(SET CMP0057 NEW)
set(LONG_TESTS mnist inferenceModel)

function(add_poptorch_py_user_guide_example name path)
  message(STATUS "Adding python example '${name}'")
  set(extra_labels "")
  if("${name}" STREQUAL "pipeline_simple")
    set(extra_labels ";external_data")
  else()
    if("${name}" IN_LIST LONG_TESTS)
      set(extra_labels "")
    else()
      set(extra_labels ";short")
    endif()
  endif()

  add_test(NAME "${name}_user_guide_example"
           COMMAND python3 ${path}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_tests_properties("${name}_user_guide_example" PROPERTIES LABELS "user_guide_examples${extra_labels}")
endfunction()

file(GLOB EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
if(COPY_TESTS)
  install(FILES ${EXAMPLES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
  set(DOC_EXAMPLES_PATH "${CMAKE_CURRENT_BINARY_DIR}")
else()
  set(DOC_EXAMPLES_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

foreach(EXAMPLE ${EXAMPLES})
  get_filename_component(NAME ${EXAMPLE} NAME_WE)
  add_poptorch_py_user_guide_example(${NAME} ${DOC_EXAMPLES_PATH})
endforeach()

if(BUILD_DOCS)
  run_poptorch_install_command(
    "python3 ${PROJECT_SOURCE_DIR}/scripts/docs_build.py --install-dir ${CMAKE_INSTALL_PREFIX} --add-to-sys-path ${CMAKE_INSTALL_PREFIX}"
    "${PROJECT_BINARY_DIR}"
    "docs_build.py")
endif()
