find_package(DRC)
if(HAVE_DRC)
    include_directories(${DRC_INCLUDE_DIRS})
    add_definitions(-DHAVE_DRC)
endif()

# default: No CUDA
set(HAVE_CUDA FALSE)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        add_definitions(-DHAVE_CUDA)
        set(HAVE_CUDA TRUE)
    endif()
else()
    find_package(CUDA)
    if(CUDA_FOUND)
        enable_language(CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        add_definitions(-DHAVE_CUDA)
        set(HAVE_CUDA TRUE)
    endif()
endif()

# list of source files
set(dspaces-src util.c bbox.c ss_data.c dspaces-client.c dspaces-ops.c)
if(HAVE_CUDA)
    list(APPEND dspaces-src  ssd_copy.cu ss_data_cuda.c)
    set_source_files_properties(ssd_copy.cu PROPERTIES LANGUAGE "CUDA")
endif()

# load package helper for generating cmake CONFIG packages
include (CMakePackageConfigHelpers)

# where to install files for "find_package"
set (dspaces-pkg "share/cmake/dspaces")

set (dspaces-vers "${dspaces_VERSION_MAJOR}.${dspaces_VERSION_MINOR}")

add_library(dspaces ${dspaces-src})
if(HAVE_CUDA)
    set_target_properties(dspaces PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# We need both cuda driver and cuda runtime headers and libs
# ${CUDA_LIBRARIES} returned by find_package(CUDA) only has the cuda runtime libs
# so we have to manually add -lcuda

target_link_libraries (dspaces margo m pthread liblz4)
target_include_directories (dspaces PUBLIC $<INSTALL_INTERFACE:include>)

if(HAVE_DRC)
    target_link_libraries (dspaces ${DRC_LIBRARIES})
    target_include_directories (dspaces PUBLIC ${DRC_INCLUDE_DIRS})
endif()

if(HAVE_CUDA)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
        target_link_libraries (dspaces CUDA::cudart CUDA::cuda_driver)
        target_include_directories (dspaces PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    else()
        target_link_libraries (dspaces cuda ${CUDA_LIBRARIES})
        target_include_directories (dspaces PUBLIC ${CUDA_INCLUDE_DIRS})
    endif()
endif()

# local include's BEFORE, in case old incompatable .h files in prefix/include
target_include_directories (dspaces BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# for shared libs, establish the lib version
set_target_properties (dspaces
    PROPERTIES VERSION ${dspaces_VERSION}
    SOVERSION ${dspaces_VERSION_MAJOR})

set(dspaces-server-src util.c bbox.c ss_data.c dspaces-server.c toml.c)

add_library(dspaces-server ${dspaces-server-src} ${dspaces-src})

target_link_libraries (dspaces-server margo m pthread liblz4 ${CURL_LIBRARIES})
target_include_directories (dspaces-server BEFORE PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> ${CURL_INCLUDE_DIR})

if(HAVE_DRC)
    target_link_libraries (dspaces-server ${DRC_LIBRARIES})
    target_include_directories (dspaces-server BEFORE PUBLIC ${DRC_INCLUDE_DIRS})
endif()

if(Python_FOUND)
    message(" ${PYTHON_INCLUDE_PATH} ${_Python_NumPy_INCLUDE_DIR}")
    target_link_libraries(dspaces-server Python::Python)
    target_include_directories (dspaces-server BEFORE PUBLIC ${PYTHON_INCLUDE_PATH} ${_Python_NumPy_INCLUDE_DIR})
    target_compile_definitions(dspaces-server PRIVATE DSPACES_HAVE_PYTHON)
endif()

target_compile_definitions(dspaces-server PRIVATE DSPACES_MOD_DIR=${CMAKE_INSTALL_PREFIX}/share/modules)

# for shared libs, establish the lib version
set_target_properties (dspaces-server
    PROPERTIES VERSION ${dspaces_VERSION}
    SOVERSION ${dspaces_VERSION_MAJOR})

#
# installation stuff (packaging and install commands)
#
write_basic_package_version_file(
    "dspaces-config-version.cmake"
    VERSION ${dspaces_VERSION}
    COMPATIBILITY AnyNewerVersion)

# generate our config file for find_package()
configure_file (dspaces-config.cmake.in dspaces-config.cmake @ONLY)

# some bits for the pkg-config file
set (DEST_DIR "${CMAKE_INSTALL_PREFIX}")
set (PRIVATE_LIBS "-ldspaces -lpthread -lmargo -labt -lmercury -lm")
configure_file ("dspaces.pc.in" "dspaces.pc" @ONLY)
configure_file ("dspaces-server.pc.in" "dspaces-server.pc" @ONLY)

#
# "make install" rules
#
install (TARGETS dspaces dspaces-server EXPORT dspaces-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib)
install (EXPORT dspaces-targets NAMESPACE dspaces::
         DESTINATION ${dspaces-pkg}
         FILE "dspaces-targets.cmake")
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/dspaces-config.cmake"
               "${CMAKE_CURRENT_BINARY_DIR}/dspaces-config-version.cmake"
               "../cmake/xpkg-import.cmake"
         DESTINATION ${dspaces-pkg} )
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/dspaces.pc"
		DESTINATION "lib/pkgconfig/")
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/dspaces-server.pc"
        DESTINATION "lib/pkgconfig/")   
