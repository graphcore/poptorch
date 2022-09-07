cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(poptorch)

add_library(poptorch_internal_headers INTERFACE)
target_include_directories(poptorch_internal_headers INTERFACE include source/include)

# Ensure ABI matches that of PyTorch
add_definitions(${TORCH_CXX_FLAGS})

add_library(poptorch SHARED
  "source/AddDetachOperations.cpp"
  "source/AddSubgraphConnectionNodes.cpp"
  "source/AliasProcessing.cpp"
  "source/AutomaticCasting.cpp"
  "source/CPUOffloadingCleanUp.cpp"
  "source/CanonicalizeLists.cpp"
  "source/ErrorOnUnsupportedAten.cpp"
  "source/FixupSetAvailableMemory.cpp"
  "source/GatherWithExpandedIndicesOptimization.cpp"
  "source/ImplicitCasting.cpp"
  "source/InplaceOps.cpp"
  "source/LowerToPopart.cpp"
  "source/LowerToPopartFactories.cpp"
  "source/OpBuilder.cpp"
  "source/OverlappedIO.cpp"
  "source/PopartCanonicalization.cpp"
  "source/PopartLateCanonicalization.cpp"
  "source/PoplarExecutable.cpp"
  "source/PoptorchSymbols.cpp"
  "source/RemoveSurplusIdentityLosses.cpp"
  "source/ScatterAddOptimization.cpp"
  "source/SessionOptionsParser.cpp"
  "source/Utils.cpp"
  "source/popart_canonicalization/ActivationOps.cpp"
  "source/popart_canonicalization/ArithmeticOps.cpp"
  "source/popart_canonicalization/AtenHandlers.gen.cpp"
  "source/popart_canonicalization/BilinearOps.cpp"
  "source/popart_canonicalization/BitwiseOps.cpp"
  "source/popart_canonicalization/BlasOps.cpp"
  "source/popart_canonicalization/ConstantOps.cpp"
  "source/popart_canonicalization/ConvolutionOps.cpp"
  "source/popart_canonicalization/CustomOps.cpp"
  "source/popart_canonicalization/DistanceOps.cpp"
  "source/popart_canonicalization/DropoutOps.cpp"
  "source/popart_canonicalization/EinsumOp.cpp"
  "source/popart_canonicalization/EmbeddingOps.cpp"
  "source/popart_canonicalization/IndexOps.cpp"
  "source/popart_canonicalization/LossOps.cpp"
  "source/popart_canonicalization/NormalizationOps.cpp"
  "source/popart_canonicalization/OtherOps.cpp"
  "source/popart_canonicalization/PoolingOps.cpp"
  "source/popart_canonicalization/PopartCanonicalizationUtils.cpp"
  "source/popart_canonicalization/PoptorchHandlers.gen.cpp"
  "source/popart_canonicalization/PyGTorchScatterOps.cpp"
  "source/popart_canonicalization/RNNOps.cpp"
  "source/popart_canonicalization/RandomSamplingOps.cpp"
  "source/popart_canonicalization/ReduceOps.cpp"
  "source/popart_canonicalization/ReshapeOps.cpp"
  "source/popart_canonicalization/SliceOps.cpp"
  "source/popart_canonicalization/SoftmaxOps.cpp"
  "source/popart_canonicalization/TensorOps.cpp"
  "source/type_and_constant_canonicalization/AddListNumElements.cpp"
  "source/type_and_constant_canonicalization/CanonicaliseConstants.cpp"
  "source/type_and_constant_canonicalization/CanonicalizeHalf.cpp"
  "source/type_and_constant_canonicalization/CastUnsupportedInputs.cpp"
  "source/type_and_constant_canonicalization/CheckAndChangeOutputTypes.cpp"
  "source/type_and_constant_canonicalization/EvaluateConstexprs.cpp"
  "source/type_and_constant_canonicalization/MakeConstantIntParams.cpp"
)

file(GLOB_RECURSE poptorch_public_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp*")

set_target_properties(poptorch PROPERTIES
  CXX_STANDARD 17
  PUBLIC_HEADER "${poptorch_public_headers}")

target_link_libraries(poptorch PUBLIC
                               popart_compiler_types
                               PRIVATE
                               dispatch_tracer
                               popart_compiler
                               poptorch_logging
                               torch
                               stdc++fs
                               )

target_include_directories(poptorch
                           PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                           PRIVATE
                           source/include)


install(TARGETS poptorch
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/poptorch
  )
