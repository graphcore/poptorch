set(WhatToDoString "Set POPART_INSTALL_DIR, \
something like -DPOPART_INSTALL_DIR=/path/to/build/install/")

FIND_PATH(POPART_INCLUDE_DIR 
  NAMES popart/builder.hpp
  HINTS ${POPART_INSTALL_DIR} ${POPART_INSTALL_DIR}/include 
  PATH_SUFFIXES popart popart/include
  DOC "directory with popart include files (popart/builder.hpp etc.)")
IF(NOT POPART_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Could not set POPART_INCLUDE_DIR. ${WhatToDoString}")
ENDIF()
MESSAGE(STATUS "Found POPART_INCLUDE_DIR ${POPART_INCLUDE_DIR}")
MARK_AS_ADVANCED(POPART_INCLUDE_DIR)

FIND_LIBRARY(POPART_LIB
  NAMES popart
  HINTS ${POPART_INSTALL_DIR}/popart/lib ${POPART_INSTALL_DIR}/lib
  PATH_SUFFIXES popart popart/lib
  DOC "popart library to link to")
IF(NOT POPART_LIB)
  MESSAGE(FATAL_ERROR "Could not set POPART_LIB. ${WhatToDoString}")
ENDIF()
MESSAGE(STATUS "Found POPART_LIB ${POPART_LIB}")
MARK_AS_ADVANCED(POPART_LIB)
