include(GNUInstallDirs)

find_package(pybind11 REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Ensure ABI matches that of PyTorch
add_definitions(${TORCH_CXX_FLAGS})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# Workaround for clang / pybind / c++17
# https://github.com/pybind/pybind11/issues/1818
add_compile_options(-fsized-deallocation)
endif()

set(POPTORCH_CORE_SOURCES
  "poptorch.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ImplicitCasting.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/LowerToPopart.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/OpBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ActivationOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/AtenHandlers.gen.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ArithmeticOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/BilinearOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/BitwiseOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/BlasOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ConstantOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ConvolutionOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/CustomOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/DistanceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/EinsumOp.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/EmbeddingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/FeatureDropoutOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/IndexOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/LossOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/LSTMOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/NormalizationOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/OtherOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PoolingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PopartCanonicalizationUtils.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/PoptorchHandlers.gen.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/RandomSamplingOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ReduceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/ReshapeOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/SliceOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/SoftmaxOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/popart_canonicalization/TensorOps.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/AddSubgraphConnectionNodes.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/CanonicalizeLists.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PopartLateCanonicalization.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoplarExecutable.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/PoptorchSymbols.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/RemoveSurplusIdentityLosses.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/ShapeInference.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CanonicaliseConstants.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CastUnsupportedInputs.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CanonicalizeHalf.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/CheckAndChangeOutputTypes.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/EvaluateConstexprs.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/type_and_constant_canonicalization/MakeConstantIntParams.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Peephole.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/EliminateListConstructs.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/Utils.cpp"
  "${PROJECT_SOURCE_DIR}/poptorch/source/WarnOnUnsupportedAten.cpp"
  )

pybind11_add_module(poptorch_core SHARED SYSTEM
  ${POPTORCH_CORE_SOURCES}
  )

target_link_libraries(poptorch_core PRIVATE
  torch
  ${TORCH_PYTHON_LIBRARY}
  popart_compiler
  poptorch_logging)

target_include_directories(poptorch_core PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/poptorch_logging/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include")

install(TARGETS poptorch_core
  DESTINATION ${INSTALL_PYDIR})

run_poptorch_install_command("python3 ${PROJECT_SOURCE_DIR}/scripts/set_version.py --torch-version ${TORCH_VERSION} ${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${PROJECT_SOURCE_DIR}" "Generate __init__.py")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION "${INSTALL_PYDIR}")
install(FILES _impl.py _options_config.py _options_impl.py _logging.py _utils.py enums.py optim.py ops.py options.py profiling.py py.typed testing.py DESTINATION "${INSTALL_PYDIR}")
