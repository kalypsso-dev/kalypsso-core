# gnu compatibility, see https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html this is used
# to retrieve portable install paths e.g. some platforms install libraries in lib, others in lib64
include(GNUInstallDirs)

if(NOT DEFINED TOP_PROJECT_NAME)
  message(FATAL_ERROR "Variable TOP_PROJECT_NAME is not defined.")
endif()

#
# if we build P4EST/SC, it also currently builds ZLIB, so we need to export ZLIB_ROOT
#
if(${TOP_PROJECT_NAME}_P4EST_BUILD)
  set(MODULEFILE_ZLIB_CONFIG "setenv ZLIB_ROOT ${CMAKE_INSTALL_PREFIX}")
endif()

set(MODULEFILE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${TOP_PROJECT_NAME_LC}-dependencies"
)
file(MAKE_DIRECTORY ${MODULEFILE_DIR})
configure_file(cmake/modulefile.in "${MODULEFILE_DIR}/1.0")

# install( FILES modulefile DESTINATION
# "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${TOP_PROJECT_NAME_LC}-dependencies"
# RENAME "1.0")
