string(REGEX REPLACE "^.*(lib/.*)$" "\\1" CMAKE_INSTALL_PYTHONDIR_DEFAULT "${Python_SITEARCH}")

set(CMAKE_INSTALL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR_DEFAULT}"
        CACHE PATH "Install directory for python modules"
     ) 
mark_as_advanced(CMAKE_INSTALL_PYTHONDIR)
set(CMAKE_PYTHON_OUTPUT_DIRECTORY
  ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}
)

include_directories(SYSTEM ${PYTHON_INCLUDE_PATH} ${_Python_NumPy_INCLUDE_DIR} ${MPI4Py_INCLUDE_DIR})
include_directories($<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

if(SWIG_FOUND)
  message("-- SWIG found: ${SWIG_EXECUTABLE}")
endif()

swig_add_library(dspaces_wrapper
                LANGUAGE python
                SOURCES dspaces_wrapper.i dspaces_wrapper.c
                )
set_target_properties(dspaces_wrapper PROPERTIES
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
swig_link_libraries(dspaces_wrapper dspaces dspaces-server)

set(CMAKE_PYTHON_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bindings/python")

install(TARGETS dspaces_wrapper
  DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/dspaces
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_PYTHON_OUTPUT_DIRECTORY}/dspaces_wrapper.py
    ${CMAKE_CURRENT_SOURCE_DIR}/dspaces.py
  DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/dspaces
)

