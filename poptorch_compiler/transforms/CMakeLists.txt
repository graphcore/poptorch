add_library(poptorch_mlir_passes STATIC
        source/CanonicalizeInplaceCopies.cpp
        source/RemoveUnusedOperations.cpp
        source/Elementwise.cpp
        )

target_link_libraries(poptorch_mlir_passes PRIVATE
        MLIRIR
        MLIRPass
        poptorch_logging
)

remove_shared_llvm(poptorch_mlir_passes)
