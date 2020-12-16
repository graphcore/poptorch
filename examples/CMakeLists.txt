function(add_poptorch_py_example name path)
  message(STATUS "Adding python example '${name}'")

  set(extra_labels "")
  if("${name}" STREQUAL "bert_ipu")
    set(extra_labels ";external_data")
  endif()
  add_test(NAME "${name}_example"
           COMMAND python3 ${path}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_tests_properties("${name}_example" PROPERTIES LABELS "examples;short${extra_labels}")
endfunction()

file(GLOB EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
if(COPY_TESTS)
  install(FILES ${EXAMPLES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
  set(EXAMPLES_PATH "${CMAKE_CURRENT_BINARY_DIR}")
else()
  set(EXAMPLES_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

foreach(EXAMPLE ${EXAMPLES})
  get_filename_component(NAME ${EXAMPLE} NAME_WE)
  add_poptorch_py_example(${NAME} ${EXAMPLES_PATH})
endforeach()
