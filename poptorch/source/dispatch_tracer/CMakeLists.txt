
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(dispatch_tracer STATIC
  RegisterAtenOverloads.cpp
  CommonHelperFunctions.cpp
  dispatchers/IDispatch.cpp
  dispatchers/JitDispatch.cpp
  ValueMapper.cpp
  Tensor.cpp
  TypeInferenceHandler.cpp
)

target_link_libraries(dispatch_tracer
  PUBLIC
    torch
  PRIVATE
    poptorch_internal_headers
    poptorch_logging
    poptorch_compiler
    popart_compiler
    poptorch_err
)

set_property(TARGET dispatch_tracer PROPERTY CXX_STANDARD 17)
