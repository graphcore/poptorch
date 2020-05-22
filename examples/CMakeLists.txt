function(add_poptorch_py_example name)
  message(STATUS "Adding python example '${name}'")

  add_test(NAME "${name}_example"
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_poptorch_py_example(simple_adder)
