cmake_minimum_required(VERSION 3.18)

project(KalypssoHDF5ReaderPlugin C CXX Fortran)

include(GNUInstallDirs)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

find_package(ParaView REQUIRED)

if(PARAVIEW_USE_MPI)
  find_package(MPI)
endif()

paraview_plugin_scan(
  PLUGIN_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/paraview.plugin"
  PROVIDES_PLUGINS
  plugins
  REQUIRES_MODULES
  required_modules
  ENABLE_BY_DEFAULT
  ON)

foreach(module IN LISTS required_modules)
  if(NOT TARGET "${module}")
    message("Missing required module: ${module}")
    return()
  endif()
endforeach()

paraview_plugin_build(
  RUNTIME_DESTINATION
  "${CMAKE_INSTALL_BINDIR}"
  LIBRARY_DESTINATION
  "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY_SUBDIRECTORY
  "${PARAVIEW_PLUGIN_SUBDIR}"
  PLUGINS
  ${plugins})
