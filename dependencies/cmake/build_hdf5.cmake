#
# Option to use git repository (instead of tarball release) for downloading Hdf5
#
option(${TOP_PROJECT_NAME}_HDF5_USE_GIT
       "Turn ON if you want to use git to download HDF5 sources (default: OFF)" OFF)

option(${TOP_PROJECT_NAME}_HDF5_USE_AUTOTOOLS
       "Turn ON if you want to build hdf5 with autotools (default: OFF)" OFF)

set(CMAKE_BUILD_TYPE_HDF5
    "Release"
    CACHE STRING "hdf5 build type")
if(NOT CMAKE_BUILD_TYPE_HDF5)
  message(STATUS "Setting hdf5 build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE_HDF5
      "Release"
      CACHE STRING
            "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
            FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE_HDF5 PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                                    "RelWithDebInfo")
endif()

#
# check if user requested a build of HDF5
#
if(${TOP_PROJECT_NAME}_HDF5_BUILD)

  message("[${TOP_PROJECT_NAME} / HDF5] Building HDF5 from source")

  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external/hdf5)

  # set install path
  set(HDF5_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

  #
  include(ExternalProject)

  # gnu compatibility, see https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html this is
  # used to retrieve portable install paths e.g. some platforms install libraries in lib, others in
  # lib64
  include(GNUInstallDirs)

  # set minimal autotools options
  set(AUTOTOOLS_HDF5_ARGS
      --prefix=${HDF5_INSTALL_PREFIX}
      --with-pthread
      --enable-linux-lfs
      --enable-shared
      --enable-build-mode=production
      --disable-sharedlib-rpath
      --with-zlib
      --with-default-api-version=v18
      --with-szlib)

  # append options for a MPI/parallel-hdf5 build
  list(
    APPEND
    AUTOTOOLS_HDF5_ARGS
    --enable-unsupported
    --enable-cxx
    --enable-parallel=yes
    CC=mpicc
    CXX=mpicxx
    FC=mpif90
    F9X=mpif90)

  # set minimal cmake options
  set(CMAKE_HDF5_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${HDF5_INSTALL_PREFIX}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE_HDF5} -DDEFAULT_API_VERSION=v18
      -DHDF5_ENABLE_PARALLEL=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=ON)

  #
  # there are 2 ways to retrieve HDF5 sources:
  #
  # 1. Use git to perform a local clone of the github HDF5 repository
  # 2. Use an archive file, variable ${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE is used to specify the
  #    full path or remote URL to that archive, this variable can be set on the cmake command line,
  #    By default we download
  #    https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.9/src/hdf5-1.10.9.tar.gz
  #    But you can also use a local file, just add
  #    `-D${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE=$HOME/hdf5-x.y.z.tar.gz` on the configure command
  #    line

  # define the default HDF5 version used to download source from github
  if(NOT DEFINED DEP_${TOP_PROJECT_NAME}_HDF5_USE_GIT_HDF5_VERSION)
    set(DEP_${TOP_PROJECT_NAME}_HDF5_USE_GIT_HDF5_VERSION 1.10.9)
  endif()

  #
  # source archive setup: git repository or archive file (remote or local)
  #
  if(NOT DEFINED ${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE)
    set(DEP_${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE
        https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.9/src/hdf5-1.10.9.tar.gz
        CACHE STRING "HDF5 source archive (URL or local filepath).")
  else()
    if(NOT EXISTS "${${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE}")
      message(
        WARNING
          "HDF5 source archive ${${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE} doesn't exist on local filesystem. If you manually specified a remote URL or use GIT, you can ignore this warning."
      )
    endif()
  endif()

  message(
    STATUS "HDF5 build with mpi support using autotools command line : ${AUTOTOOLS_HDF5_ARGS}")

  # if we chose to build p4est, let hdf5 depends upon it (because currently p4est also builds zlib,
  # so we want to use that version of zlib)
  set(DEPENDENCIES_HDF5)
  if(${TOP_PROJECT_NAME}_P4EST_BUILD)
    list(APPEND DEPENDENCIES_HDF5 p4est_external)
  endif()

  #
  # By default use cmake build
  #
  if(${TOP_PROJECT_NAME}_HDF5_USE_AUTOTOOLS)

    if(${TOP_PROJECT_NAME}_HDF5_USE_GIT)
      message(
        "Building HDF5 using github repository with tag version ${DEP_${TOP_PROJECT_NAME}_HDF5_USE_GIT_HDF5_VERSION}"
      )
      ExternalProject_Add(
        hdf5_external
        DEPENDS ${DEPENDENCIES_HDF5}
        GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
        GIT_TAG ${DEP_${TOP_PROJECT_NAME}_HDF5_USE_GIT_HDF5_VERSION}
        UPDATE_DISCONNECTED true
        CONFIGURE_HANDLED_BY_BUILD true
        CONFIGURE_COMMAND touch <SOURCE_DIR>/.abi_stamp
        COMMAND autoreconf -i <SOURCE_DIR>
        COMMAND <SOURCE_DIR>/configure ${AUTOTOOLS_HDF5_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
    else()
      message("Building HDF5 using source archive ${${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE}")
      ExternalProject_Add(
        hdf5_external
        URL ${${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE}
        UPDATE_DISCONNECTED true
        CONFIGURE_HANDLED_BY_BUILD true
        CONFIGURE_COMMAND <SOURCE_DIR>/configure ${AUTOTOOLS_HDF5_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
    endif()

  else()
    # use cmake
    if(${TOP_PROJECT_NAME}_HDF5_USE_GIT)
      message(
        "Building HDF5 using github repository with tag version ${DEP_${TOP_PROJECT_NAME}_HDF5_USE_GIT_HDF5_VERSION}"
      )
      ExternalProject_Add(
        hdf5_external
        DEPENDS ${DEPENDENCIES_HDF5}
        GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
        GIT_TAG ${DEP_${TOP_PROJECT_NAME}_HDF5_USE_GIT_HDF5_VERSION}
        UPDATE_DISCONNECTED true
        CMAKE_ARGS ${CMAKE_HDF5_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
    else()
      message("Building HDF5 using source archive ${${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE}")
      ExternalProject_Add(
        hdf5_external
        URL ${${TOP_PROJECT_NAME}_HDF5_SOURCE_ARCHIVE}
        UPDATE_DISCONNECTED true
        CMAKE_ARGS ${CMAKE_HDF5_ARGS}
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j 8
        LOG_DOWNLOAD 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_INSTALL 1)
    endif()

  endif()

  message("[${TOP_PROJECT_NAME} / HDF5] HDF5 will be installed into ${HDF5_INSTALL_PREFIX}")

endif(${TOP_PROJECT_NAME}_HDF5_BUILD)
