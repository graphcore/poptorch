add_library(poptorch_mlir_passes STATIC
        source/CommonPasses.cpp
        source/Elementwise.cpp
        source/Reductions.cpp
        source/Reshapes.cpp
        source/TensorOps.cpp

        source/Helpers.cpp
        source/OutplaceViewOpsPass.cpp
        source/RemoveOverwritePass.cpp
        )

target_link_libraries(poptorch_mlir_passes PRIVATE
        MLIRIR
        MLIRPass
        poptorch_logging
)

target_include_directories(poptorch_mlir_passes PUBLIC include)

remove_shared_llvm(poptorch_mlir_passes)
