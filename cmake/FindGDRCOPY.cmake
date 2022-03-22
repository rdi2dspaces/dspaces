# - Try to find GDRCOPY
# Once done this will define
#  GDRCOPY_FOUND - System has GDRCOPY
#  GDRCOPY_INCLUDE_DIRS - The GDRCOPY include directories
#  GDRCOPY_LIBRARIES - The libraries needed to use GDRCOPY

find_path(GDRCOPY_INCLUDE_DIR gdrapi.h
  HINTS ${GDRCOPY_INCLUDEDIR} ${GDRCOPY_INCLUDE_DIRS})

find_library(GDRCOPY_LIBRARY NAMES gdrapi
  HINTS ${GDRCOPY_LIBDIR} ${GDRCOPY_LIBRARY_DIRS})

set(GDRCOPY_INCLUDE_DIRS ${GDRCOPY_INCLUDE_DIR})
set(GDRCOPY_LIBRARIES ${GDRCOPY_LIBRARY})
include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set DRC_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GDRCOPY DEFAULT_MSG
                                  GDRCOPY_INCLUDE_DIR GDRCOPY_LIBRARY)

mark_as_advanced(GDRCOPY_INCLUDE_DIR GDRCOPY_LIBRARY)

if(GDRCOPY_FOUND)
  set(HAVE_GDRCOPY TRUE)
endif()