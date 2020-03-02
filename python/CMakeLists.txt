include(GNUInstallDirs)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(poptorch_core SHARED
  "poptorch.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/LowerToPopart.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OpBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoplarExecutable.cpp"
  )

target_link_libraries(poptorch_core PRIVATE
  ${TORCH_LIBRARIES}
  ${POPART_LIB}
  popart_compiler)

target_include_directories(poptorch_core PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include"
  ${POPART_INCLUDE_DIR})

install(TARGETS poptorch_core
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES __init__.py DESTINATION "${INSTALL_PYDIR}/poptorch")
