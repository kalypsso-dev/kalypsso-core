#
# Option to use git (instead of tarball release) for downloading p4est
#
# Turn ON by default, to retrieve p4est and submodule sc.
#
option(${TOP_PROJECT_NAME}_P4EST_USE_GIT
       "Turn ON if you want to use git to download p4est sources (default: ON)" ON)

#
# This option is only as a "fix" because currently cmake enforce downloading zlib sources, while the
# autotools build doesn't; so for now just keep using the autotools build.
#
option(${TOP_PROJECT_NAME}_P4EST_USE_AUTOTOOLS
       "Turn ON if you want to build p4est with autotools (default: OFF)" OFF)

set(CMAKE_BUILD_TYPE_P4EST
    "Release"
    CACHE STRING "p4est build type")
if(NOT CMAKE_BUILD_TYPE_P4EST)
  message(STATUS "Setting p4est build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE_P4EST
      "Release"
      CACHE STRING
            "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
            FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE_P4EST PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                                     "RelWithDebInfo")
endif()

#
# check if user requested a build of p4est
#
if(${TOP_PROJECT_NAME}_P4EST_BUILD)

  message("[kalypsso / p4est] Building p4est from source")

  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external/p4est)

  # set install path
  set(P4EST_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

  # set minimal autotools options
  set(AUTOTOOLS_P4EST_ARGS --prefix=${P4EST_INSTALL_PREFIX} --enable-mpi --enable-mpiio
                           --enable-shared)

  # set minimal cmake options don't build zlib, just use the one from system
  set(CMAKE_P4EST_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${P4EST_INSTALL_PREFIX}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE_P4EST}
      -DBUILD_SHARED_LIBS=ON
      -DP4EST_ENABLE_BUILD_2D=ON
      -Dsc_external=ON
      -Denable_p8est=ON
      -Dvtk_binary=ON
      -Dzlib=OFF)

  find_package(MPI COMPONENTS Fortran C CXX)
  if(MPI_FOUND)
    list(APPEND CMAKE_P4EST_ARGS -Dmpi=ON)
  else()
    message(FATAL_ERROR "We want MPI when using p4est inside kalypsso.")
  endif()

  #
  # Make ExternalProject macro available
  #
  include(ExternalProject)

  # gnu compatibility, see https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html this is
  # used to retrieve portable install paths e.g. some platforms install libraries in lib, others in
  # lib64
  include(GNUInstallDirs)

  #
  # there are 2 ways to retrieve p4est sources:
  #
  # 1. Use git to perform a local clone of the github p4est repository
  # 2. Use an archive file, variable DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE is used to specify
  #    the full path or remote URL to that archive. This variable can be set on the cmake command
  #    line, e.g. `-DDEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE=$HOME/p4est-2.8.5.tar.gz` Be
  #    careful to provide a tarball with p4est and sc sources.

  # define the default p4est version used to download source from github
  if(NOT DEFINED DEP_${TOP_PROJECT_NAME}_P4EST_USE_GIT_P4EST_VERSION)
    set(DEP_${TOP_PROJECT_NAME}_P4EST_USE_GIT_P4EST_VERSION 2.8.5)
  endif()

  #
  if(NOT DEFINED DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE)
    set(DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE
        https://p4est.github.io/release/p4est-2.8.5.5-9ddbb.tar.gz
        CACHE STRING "p4est source archive (URL or local filepath).")
  else()
    if(NOT EXISTS "${DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE}")
      message(
        WARNING
          "P4EST source archive ${DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE} doesn't exist on local filesystem. If you manually specified a remote URL or use GIT, you can ignore this warning."
      )
    endif()
  endif()

  message(STATUS "p4est build with cmake command line : ${CMAKE_P4EST_ARGS}")

  #
  # By default use autotools build
  #
  if(${TOP_PROJECT_NAME}_P4EST_USE_AUTOTOOLS)
    # use Autotools
    if(${TOP_PROJECT_NAME}_P4EST_USE_GIT)
      message(
        "Building p4est with autotools using github repository with tag version ${DEP_${TOP_PROJECT_NAME}_P4EST_USE_GIT_P4EST_VERSION}"
      )
      # cmake-format: off
      ExternalProject_Add(
        p4est_external
        # GIT_REPOSITORY https://github.com/cburstedde/p4est.git
        # GIT_TAG v${DEP_${TOP_PROJECT_NAME}_P4EST_USE_GIT_P4EST_VERSION}
        GIT_REPOSITORY https://github.com/pkestene/p4est.git
        GIT_TAG 2.8.5-ahead
        UPDATE_DISCONNECTED true
        CONFIGURE_HANDLED_BY_BUILD true
        CONFIGURE_COMMAND touch <SOURCE_DIR>/.p4est_stamp
        COMMAND ./bootstrap
        COMMAND <SOURCE_DIR>/configure ${AUTOTOOLS_P4EST_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        BUILD_IN_SOURCE 1
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
      # cmake-format: on
    else()
      message(
        "Building p4est with autotools using source archive ${DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE}"
      )
      ExternalProject_Add(
        p4est_external
        URL ${DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE} CONFIGURE_HANDLED_BY_BUILD true
        CONFIGURE_COMMAND touch <SOURCE_DIR>/.p4est_stamp
        COMMAND ./bootstrap
        COMMAND <SOURCE_DIR>/configure ${AUTOTOOLS_P4EST_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        BUILD_IN_SOURCE 1
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
    endif()

  else()
    # use CMAKE
    if(${TOP_PROJECT_NAME}_P4EST_USE_GIT)
      message(
        "Building p4est with cmake using github repository with tag version ${DEP_${TOP_PROJECT_NAME}_P4EST_USE_GIT_P4EST_VERSION}"
      )
      # cmake-format: off
      ExternalProject_Add(
        p4est_external
        # GIT_REPOSITORY https://github.com/cburstedde/p4est.git
        # GIT_TAG v${DEP_${TOP_PROJECT_NAME}_P4EST_USE_GIT_P4EST_VERSION}
        GIT_REPOSITORY https://github.com/pkestene/p4est.git
        GIT_TAG 2.8.5-ahead
        CMAKE_ARGS ${CMAKE_P4EST_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
      # cmake-format: on
    else()
      message(
        "Building p4est with cmake using source archive ${DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE}"
      )
      ExternalProject_Add(
        p4est_external
        URL ${DEP_${TOP_PROJECT_NAME}_P4EST_SOURCE_ARCHIVE}
        CMAKE_ARGS ${CMAKE_P4EST_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
    endif()
  endif()

  message("[kalypsso / p4est] p4est will be installed into ${P4EST_INSTALL_PREFIX}")

endif(${TOP_PROJECT_NAME}_P4EST_BUILD)
