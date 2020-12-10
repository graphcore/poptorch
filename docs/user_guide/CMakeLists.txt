function(add_poptorch_py_user_guide_example name)
  message(STATUS "Adding python example '${name}'")
  set(extra_labels "")
  if("${name}" STREQUAL "pipeline_simple")
    set(extra_labels ";external_data")
  endif()

  add_test(NAME "${name}_user_guide_example"
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
  set_tests_properties("${name}_user_guide_example" PROPERTIES LABELS "user_guide_examples;short${extra_labels}")
endfunction()

file(GLOB EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(EXAMPLE ${EXAMPLES})
  get_filename_component(NAME ${EXAMPLE} NAME_WE)
  add_poptorch_py_user_guide_example(${NAME})
endforeach()

if(BUILD_DOCS)
  find_package(poplar REQUIRED)
  find_package(popart REQUIRED COMPONENTS popart-only)

  get_target_property(POPLAR_LIB poplar LOCATION)
  get_filename_component(POPLAR_DIR ${POPLAR_LIB} DIRECTORY)
  install(CODE
    "set(ENV{LD_LIBRARY_PATH} ${popart_LIB_DIR}:${POPLAR_DIR}:$ENV{LD_LIBRARY_PATH})
    execute_process( COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/docs_build.py --install-dir ${CMAKE_INSTALL_PREFIX} --add-to-sys-path ${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY ${PROJECT_BINARY_DIR} RESULT_VARIABLE RETVAL )
    if(\${RETVAL})
      message(FATAL_ERROR \"docs_build.py FAILED (See above)\")
    endif()")
endif()
