include(GNUInstallDirs)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# Ensure ABI matches that of PyTorch
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=${Torch_USE_CXX11_ABI}")

set(POPTORCH_CORE_SOURCES
  "poptorch.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/LowerToPopart.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OpBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ArithmeticOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ComparisonOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ConstantOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/EmbeddingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/FillOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/LogOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/LossOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/LSTMOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/NormalizationOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PoolingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PopartCanonicalizationUtils.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ReduceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ReshapeOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ScopeOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/SoftmaxOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ConvolutionOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/BilinearOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/CanonicalizeLists.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartLateCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoplarExecutable.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ShapeInference.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CastUnsupportedInputs.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Peephole.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/EliminateListConstructs.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Utils.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/WarnOnUnsupportedAten.cpp"
  )

pybind11_add_module(poptorch_core SHARED SYSTEM
  ${POPTORCH_CORE_SOURCES}
  )

target_link_libraries(poptorch_core PRIVATE
  ${TORCH_LIBRARIES}
  popart_compiler
  poptorch_logging)

target_include_directories(poptorch_core PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/poptorch_logging/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include")

target_include_directories(poptorch_core SYSTEM PRIVATE
  ${SPDLOG_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS})

install(TARGETS poptorch_core
  DESTINATION ${INSTALL_PYDIR})

install(FILES __init__.py testing.py DESTINATION "${INSTALL_PYDIR}")
