# FindP4EST.cmake
# ---------------
#
# Try to find p4est library (use env variable P4EST_ROOT as a hint. If libsc is installed in a
# different location than p4est, you can use SC_ROOT to provide an hint)
#
# Recommendation: if you use module-environment, just make sure variable P4EST_ROOT is set to
# top-level directory where p4est was install.
#
# Result Variables
# ----------------
#
# This module defines the following variables::
#
# * P4EST_FOUND          - True if P4EST was found
# * P4EST_INCLUDE_DIRS   - include directories for P4EST
# * P4EST_LIBRARIES      - link against this library to use P4EST
#
# The module will also define two cache variables::
#
# * P4EST_INCLUDE_DIR    - the P4EST include directory
# * P4EST_LIBRARY        - the path to the P4EST library
#
# This module also exports the following target:: p4est

if(NOT P4EST_ROOT)
  if(DEFINED ENV{P4EST_ROOT})
    set(SC_ROOT $ENV{P4EST_ROOT})
  else()
    message(STATUS "(WARNING) Env variable P4EST_ROOT not defined...")
  endif()
endif()

find_path(
  P4EST_INCLUDE_DIR
  NAMES p4est.h
  PATH_SUFFIXES include
  HINTS ${P4EST_DIR} ${P4EST_ROOT})
mark_as_advanced(P4EST_INCLUDE_DIR)

find_library(
  P4EST_LIBRARY
  NAMES p4est
  PATH_SUFFIXES lib
  HINTS ${P4EST_DIR} ${P4EST_ROOT})
mark_as_advanced(P4EST_LIBRARY)

# try to detect P4EST version
if(P4EST_INCLUDE_DIR AND EXISTS "${P4EST_INCLUDE_DIR}/p4est_config.h")
  file(STRINGS "${P4EST_INCLUDE_DIR}/p4est_config.h" P4EST_H
       REGEX "^#define P4EST_VERSION \"[^\"]*\"$")

  # try to extract the string inside quote
  string(REGEX REPLACE "^.*P4EST_VERSION \"([^\"]+).*\"$" "\\1" P4EST_VERSION_STRING "${P4EST_H}")

else()
  set(P4EST_VERSION_STRING "unknown")
endif()
mark_as_advanced(P4EST_VERSION_STRING)

# handle the QUIETLY and REQUIRED arguments and set P4EST_FOUND to TRUE if all listed variables are
# TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  P4EST
  REQUIRED_VARS P4EST_LIBRARY P4EST_INCLUDE_DIR
  VERSION_VAR P4EST_VERSION_STRING)

if(P4EST_FOUND)
  set(P4EST_INCLUDE_DIRS ${P4EST_INCLUDE_DIR})
endif(P4EST_FOUND)

if(P4EST_FOUND AND NOT TARGET p4est::p4est)
  add_library(p4est::p4est SHARED IMPORTED)

  # We need to link libsc, so we look for it
  find_package(SC REQUIRED)

  target_link_libraries(p4est::p4est INTERFACE sc::sc)

  set_target_properties(
    p4est::p4est PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${P4EST_INCLUDE_DIRS}"
                            IMPORTED_LOCATION "${P4EST_LIBRARY}")
endif()
