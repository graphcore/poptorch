function(add_poptorch_py_unit_test name)
  message(STATUS "Adding python test '${name}'")

  add_test(NAME "${name}_test"
           COMMAND python3 -m pytest -s ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
endfunction()

add_poptorch_py_unit_test(shape_inference)
