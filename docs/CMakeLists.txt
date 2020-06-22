cmake_minimum_required(VERSION 3.4.2)

set(BUILD_DOCS OFF CACHE BOOL "Build documentation")

if (BUILD_DOCS)
  find_package(gcdoc REQUIRED)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND cmake -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Creating build folder for 'docs'"
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/latex
    COMMAND cmake -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/latex
    COMMENT "Creating build folder for 'docs/latex'"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pdfgen
    COMMAND ${GCDOC} -v -r ${DOCS_VERSION} --onesided -b pdf ${CMAKE_CURRENT_SOURCE_DIR}/user_guide ${CMAKE_CURRENT_BINARY_DIR}/latex/poptorch_user_guide.pdf
    COMMENT "Building PDF documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/poptorch_user_guide
    COMMAND ${GCDOC} -v -r ${DOCS_VERSION} -b singlehtml ${CMAKE_CURRENT_SOURCE_DIR}/user_guide ${CMAKE_CURRENT_BINARY_DIR}/html/poptorch_user_guide
    COMMENT "Building html user guide documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(ug_zip ALL
    COMMAND zip -r ${PROJECT_BINARY_DIR}/${USER_GUIDE_HTML_NAME} . -i *
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/poptorch_user_guide
    VERBATIM
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/poptorch_user_guide
  )

  add_custom_target(pdf_copy ALL
    COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/latex/poptorch_user_guide.pdf ${PROJECT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/pdfgen
  )

endif()
