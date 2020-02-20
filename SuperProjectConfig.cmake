set(Torch_DIR "" CACHE STRING "The torch install dir")
list(APPEND POPTORCH_CMAKE_ARGS -DTorch_DIR=${Torch_DIR})
