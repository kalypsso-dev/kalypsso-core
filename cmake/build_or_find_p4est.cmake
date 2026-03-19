#
# find or build p4est
#

# default value is OFF
option(KALYPSSO_CORE_BUILD_P4EST "Building p4est" OFF)

# build p4est ?
if(KALYPSSO_CORE_BUILD_P4EST)

  message("[kalypsso-core] Building p4est from source")

  set(DEFAULT_BUILD_TYPE_P4EST "Release")
  if(NOT CMAKE_BUILD_TYPE_P4EST)
    message(STATUS "Setting p4est build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE_P4EST
        "${DEFAULT_BUILD_TYPE_P4EST}"
        CACHE
          STRING
          "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
          FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE_P4EST PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                                       "RelWithDebInfo")
  endif()

  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external/p4est)

  # gnu compatibility, see https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html this is
  # used to retrieve portable install paths e.g. some platforms install libraries in lib, others in
  # lib64
  include(GNUInstallDirs)

  # enforce a local build of p4est
  set(P4EST_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/external/install/p4est)
  set(P4EST_INCLUDEDIR ${P4EST_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})

  #
  # note about p4est build system:
  #
  # * old versions p4est were not following GnuInstallDirs standards. p4est was always selecting to
  #   install libraries in lib, whereas GnuInstallDirs will chose to install in a directory which OS
  #   dependent, e.g. lib64 on RedHat like system and lib in Debian-like systems.
  #
  # * newer p4est release v2.8.7, p4est is not using GnuInstallDirs standard
  #
  set(P4EST_LIBDIR ${P4EST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
  # set(P4EST_LIBDIR ${P4EST_INSTALL_PREFIX}/lib)

  # set minimal cmake options don't build zlib, just use the one from system
  set(CMAKE_P4EST_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${P4EST_INSTALL_PREFIX}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE_P4EST}
      -DBUILD_SHARED_LIBS=ON
      -DP4EST_ENABLE_BUILD_2D=ON
      -DP4EST_USE_SYSTEM_SC=OFF
      -Denable_p8est=ON
      -Dvtk_binary=ON
      -Dzlib=OFF)

  if(KALYPSSO_CORE_USE_MPI)
    find_package(MPI COMPONENTS C CXX)
    if(MPI_FOUND)
      list(APPEND CMAKE_P4EST_ARGS -DSC_ENABLE_MPI=ON)
    else(MPI_FOUND)
      message(FATAL_ERROR "MPI was requested but not found.")
    endif(MPI_FOUND)
  else(KALYPSSO_CORE_USE_MPI)
    list(APPEND CMAKE_P4EST_ARGS -DSC_ENABLE_MPI=OFF)
  endif(KALYPSSO_CORE_USE_MPI)

  #
  # Make ExternalProject macro available
  #
  include(ExternalProject)

  # cmake-format: off
  ExternalProject_Add(
    p4est_external
    # GIT_REPOSITORY https://github.com/cburstedde/p4est.git
    # GIT_TAG v${${TOP_PROJECT_NAME}_P4EST_USE_GIT_P4EST_VERSION}
    # GIT_REPOSITORY https://github.com/pkestene/p4est.git
    # GIT_TAG 2.8.5-ahead
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/p4est
    CMAKE_ARGS ${CMAKE_P4EST_ARGS}
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1)
  # cmake-format: on

  # work around to prevent cmake to complain about non existing path
  file(MAKE_DIRECTORY ${P4EST_INCLUDEDIR})

  #
  # need to import target, so that we can build kalypsso-core
  #

  # ##### SC #####
  add_library(sc SHARED IMPORTED GLOBAL)
  add_dependencies(sc p4est_external)

  set_target_properties(sc PROPERTIES IMPORTED_LOCATION "${P4EST_LIBDIR}/libsc.so"
                                      INTERFACE_INCLUDE_DIRECTORIES ${P4EST_INCLUDEDIR})

  # create library alias
  add_library(SC::SC ALIAS sc)

  # ##### P4EST #####
  add_library(p4est SHARED IMPORTED GLOBAL)
  add_dependencies(p4est p4est_external)

  set_target_properties(
    p4est
    PROPERTIES IMPORTED_LOCATION "${P4EST_LIBDIR}/libp4est.so"
               INTERFACE_INCLUDE_DIRECTORIES ${P4EST_INCLUDEDIR}
               INTERFACE_LINK_LIBRARIES SC::SC)

  # create library alias
  add_library(P4EST::P4EST ALIAS p4est)

  # p4est is found !
  set(P4EST_FOUND true)

  set(USE_P4EST_BUILTIN TRUE)

else(KALYPSSO_CORE_BUILD_P4EST)

  # try to detect P4EST using env variable P4EST_ROOT
  find_package(P4EST CONFIG REQUIRED)
  find_package(SC CONFIG REQUIRED)
  set(USE_P4EST_BUILTIN FALSE)

endif(KALYPSSO_CORE_BUILD_P4EST)
