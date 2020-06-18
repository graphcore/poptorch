include(GNUInstallDirs)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(Spdlog REQUIRED)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

pybind11_add_module(poptorch_core SHARED
  "poptorch.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/LowerToPopart.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OpBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoplarExecutable.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ShapeInference.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Peephole.cpp"
  "${PROJECT_SOURCE_DIR}/shared/source/Logging.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/EliminateListConstructs.cpp"
  )

target_link_libraries(poptorch_core PRIVATE
  ${TORCH_LIBRARIES}
  ${POPART_LIB}
  popart_compiler)

target_include_directories(poptorch_core PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/shared/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include"
  ${SPDLOG_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
  ${POPART_INCLUDE_DIR})

install(TARGETS poptorch_core
  DESTINATION ${INSTALL_PYDIR})

install(FILES __init__.py DESTINATION "${INSTALL_PYDIR}")
