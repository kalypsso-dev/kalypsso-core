#
# find or build cnpy (cnpy is optional); use CNPY_FOUND to appropriately enable it.
#

# default value is ON
option(KALYPSSO_CORE_BUILD_CNPY "Building cnpy" ON)

# build cnpy ?
if(KALYPSSO_CORE_BUILD_CNPY)

  # enforce a local build of cnpy
  message("[kalypsso-core] Building cnpy from source")

  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external/cnpy)

  # gnu compatibility, see https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html this is
  # used to retrieve portable install paths e.g. some platforms install libraries in lib, others in
  # lib64
  include(GNUInstallDirs)

  set(CNPY_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/external/install/cnpy)
  set(CNPY_INCLUDEDIR ${CNPY_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
  set(CNPY_LIBDIR ${CNPY_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

  # set minimal cmake options
  set(CMAKE_CNPY_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CNPY_INSTALL_PREFIX}
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_VERBOSE_BUILD:BOOL=TRUE)

  #
  # Make ExternalProject macro available
  #
  include(ExternalProject)

  # cmake-format: off
  ExternalProject_Add(
    cnpy_external
    # GIT_REPOSITORY https://github.com/pkestene/cnpy-cmake.git
    # GIT_TAG master
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/cnpy
    CMAKE_ARGS ${CMAKE_CNPY_ARGS}
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} VERBOSE=1 -j 8
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1)
  # cmake-format: on

  # work around to prevent cmake to complain about non existing path
  file(MAKE_DIRECTORY ${CNPY_INCLUDEDIR})

  #
  # need to import target, so that we can build kalypsso-core
  #

  # ##### CNPY #####
  add_library(cnpy SHARED IMPORTED GLOBAL)
  add_dependencies(cnpy cnpy_external)

  set_target_properties(cnpy PROPERTIES IMPORTED_LOCATION "${CNPY_LIBDIR}/libcnpy.so"
                                        INTERFACE_INCLUDE_DIRECTORIES ${CNPY_INCLUDEDIR})

  # create library alias
  add_library(cnpy::cnpy ALIAS cnpy)

  # cnpy is found !
  set(CNPY_FOUND true)
  set(KALYPSSO_CORE_USE_CNPY ON)
  set(USE_CNPY_BUILTIN TRUE)

else(KALYPSSO_CORE_BUILD_CNPY)

  # try to detect CNPY using env variable CNPY_ROOT
  find_package(cnpy CONFIG)

  if(cnpy_FOUND)
    set(KALYPSSO_CORE_USE_CNPY ON)
  else()
    set(KALYPSSO_CORE_USE_CNPY OFF)
  endif()

  set(USE_CNPY_BUILTIN FALSE)

endif(KALYPSSO_CORE_BUILD_CNPY)
