add_library(lower_to_poplar STATIC
        CompilerHelpers.cpp
        IMlirGraphConverter.cpp
        NonRestartingMlirTimer.cpp
        PoplarDeviceAndTarget.cpp
        PoplarExecutor.cpp
        ops/Activations.cpp
        ops/ElementWise.cpp
        ops/Linalg.cpp
        ops/Losses.cpp
        ops/NormOps.cpp
        ops/Ops.cpp
        ops/PoolingOps.cpp
        ops/Random.cpp
        ops/Reductions.cpp
        ops/Reshapes.cpp
        ops/TensorOps.cpp
        passes/LowerToPoplar.cpp)


target_include_directories(lower_to_poplar PUBLIC include)

find_package(model_runtime REQUIRED)
find_package(popef REQUIRED)

# Poplar requires exceptions and LLVM disables them for efficiency so we can't
# use the nice MLIR cmake functions here.
target_link_libraries(lower_to_poplar PRIVATE
        MLIRIR
        MLIRPass
        MLIRTransforms
        poptorch_mlir_passes

        model_runtime
        popef
        poplar
        popops
        poprand
        poplin
        popnn
        poputil
        gcl
        snap
        poptorch_logging
        )

remove_shared_llvm(lower_to_poplar)
