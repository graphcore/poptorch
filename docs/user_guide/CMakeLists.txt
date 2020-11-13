function(add_poptorch_py_user_guide_example name)
  message(STATUS "Adding python example '${name}'")

  add_test(NAME "${name}_user_guide_example"
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_tests_properties("${name}_user_guide_example" PROPERTIES LABELS "user_guide_examples;short")
endfunction()

file(GLOB EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(EXAMPLE ${EXAMPLES})
  get_filename_component(NAME ${EXAMPLE} NAME_WE)
  add_poptorch_py_user_guide_example(${NAME})
endforeach()
