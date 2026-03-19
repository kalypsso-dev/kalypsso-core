# FindSC.cmake
# ---------------
#
# Try to find sc library (use env variable SC_ROOT as a hint. If libsc is installed in a different
# location than sc, you can use SC_ROOT to provide an hint)
#
# Recommendation: if you use module-environment, just make sure variable SC_ROOT is set to top-level
# directory where sc was install.
#
# Result Variables
# ----------------
#
# This module defines the following variables::
#
# SC_FOUND          - True if SC was found SC_INCLUDE_DIRS   - include directories for SC
# SC_LIBRARIES      - link against this library to use SC
#
# The module will also define two cache variables::
#
# SC_INCLUDE_DIR    - the SC include directory SC_LIBRARY        - the path to the SC library
#
# This module also exports the following target:: sc

if(NOT SC_ROOT)
  if(DEFINED ENV{SC_ROOT})
    set(SC_ROOT $ENV{SC_ROOT})
  else()
    message(STATUS "Env variable SC_ROOT not defined...")
  endif()
endif()

find_path(
  SC_INCLUDE_DIR
  NAMES sc.h
  PATH_SUFFIXES include
  HINTS ${SC_ROOT} ${SC_DIR})
mark_as_advanced(SC_INCLUDE_DIR)

find_library(
  SC_LIBRARY
  NAMES sc
  PATH_SUFFIXES lib
  HINTS ${SC_ROOT} ${SC_DIR})
mark_as_advanced(SC_LIBRARY)

# try to detect SC version
if(SC_INCLUDE_DIR AND EXISTS "${SC_INCLUDE_DIR}/sc_config.h")
  file(STRINGS "${SC_INCLUDE_DIR}/sc_config.h" SC_H REGEX "^#define SC_VERSION \"[^\"]*\"$")

  # try to extract the string inside quote
  string(REGEX REPLACE "^.*SC_VERSION \"([^\"]+).*\"$" "\\1" SC_VERSION_STRING "${SC_H}")

else()
  set(SC_VERSION_STRING "unknown")
endif()
mark_as_advanced(SC_VERSION_STRING)

# handle the QUIETLY and REQUIRED arguments and set SC_FOUND to TRUE if all listed variables are
# TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  SC
  REQUIRED_VARS SC_LIBRARY SC_INCLUDE_DIR
  VERSION_VAR SC_VERSION_STRING)

if(SC_FOUND)
  set(SC_INCLUDE_DIRS ${SC_INCLUDE_DIR})
endif(SC_FOUND)

if(SC_FOUND AND NOT TARGET sc::sc)
  add_library(sc::sc SHARED IMPORTED)

  set_target_properties(sc::sc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SC_INCLUDE_DIR}"
                                          IMPORTED_LOCATION "${SC_LIBRARY}")
endif()
