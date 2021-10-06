
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(dispatch_tracer STATIC
        RegisterAtenOverloads.cpp
        CommonHelperFunctions.cpp
        dispatchers/JitDispatch.cpp
        ValueMapper.cpp
)

target_link_libraries(dispatch_tracer
  PUBLIC
    torch
  PRIVATE
    poptorch_logging
)

set_property(TARGET dispatch_tracer PROPERTY CXX_STANDARD 17)

target_include_directories(dispatch_tracer PRIVATE
  "${PROJECT_SOURCE_DIR}/poptorch/include"
  "${PROJECT_SOURCE_DIR}/popart_compiler/include"
)