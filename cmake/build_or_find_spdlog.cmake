#
# find or build spdlog
#
# - spdlog is not genuinely required but we strongly recommend to use it
# - if spdlog is not detected, you can always use build-in (sources available in git submodule)
#   just use cmake option \"-DKALYPSSO_CORE_BUILD_SPDLOG=ON\" to enable it
#

# default value is ON (since Kokkos 5.1.0 and using c++-20, we need to upgrade spdlog to
# version 1.17 when building for Kokkos::CUDA backend; so we can't rely on default spdlog
# version on ubuntu).
option(KALYPSSO_CORE_BUILD_SPDLOG "Building spdlog" ON)

# build spdlog ?
if(KALYPSSO_CORE_BUILD_SPDLOG)

  # enforce a local build of spdlog
  message("[kalypsso-core] Building spdlog from source")

  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external/spdlog)

  # gnu compatibility, see https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html this is
  # used to retrieve portable install paths e.g. some platforms install libraries in lib, others in
  # lib64
  include(GNUInstallDirs)

  set(SPDLOG_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/external/install/spdlog)
  set(SPDLOG_INCLUDEDIR ${SPDLOG_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
  set(SPDLOG_LIBDIR ${SPDLOG_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

  # set minimal cmake options
  set(CMAKE_SPDLOG_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${SPDLOG_INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_VERBOSE_BUILD:BOOL=TRUE -DSPDLOG_BUILD_SHARED:BOOL=ON)

  #
  # Make ExternalProject macro available
  #
  include(ExternalProject)

  ExternalProject_Add(
    spdlog_external
    # GIT_REPOSITORY https://github.com/gabime/spdlog.git GIT_TAG v1.11.0
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/spdlog
    CMAKE_ARGS ${CMAKE_SPDLOG_ARGS}
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} VERBOSE=1 -j 8
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1)

  # work around to prevent cmake to complain about non existing path
  file(MAKE_DIRECTORY ${SPDLOG_INCLUDEDIR})

  #
  # need to import target, so that we can build kalypsso-core
  #

  # ##### SPDLOG #####
  add_library(spdlog SHARED IMPORTED GLOBAL)
  add_dependencies(spdlog spdlog_external)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SPDLOG_LIBNAME "libspdlogd.so")
  else()
    set(SPDLOG_LIBNAME "libspdlog.so")
  endif()

  set_target_properties(spdlog PROPERTIES IMPORTED_LOCATION "${SPDLOG_LIBDIR}/${SPDLOG_LIBNAME}"
                                          INTERFACE_INCLUDE_DIRECTORIES ${SPDLOG_INCLUDEDIR})

  # create library alias
  add_library(spdlog::spdlog ALIAS spdlog)

  # spdlog is found !
  set(KALYPSSO_CORE_USE_SPDLOG ON)
  set(USE_SPDLOG_BUILTIN TRUE)

  # get spdlog version / tag
  find_package(Git QUIET)

  if(GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 --match=v[0-9]*
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/spdlog
      OUTPUT_VARIABLE spdlog_VERSION
      RESULT_VARIABLE spdlog_VERSION_FOUND
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT spdlog_VERSION_FOUND EQUAL 0)
      set(spdlog_VERSION "Unknown1")
    endif()

  else()

    set(spdlog_VERSION "Unknown")

  endif()

else(KALYPSSO_CORE_BUILD_SPDLOG)

  #
  # try to detect SPDLOG using env variable SPDLOG_ROOT
  #
  find_package(spdlog CONFIG)

  if(spdlog_FOUND)
    set(KALYPSSO_CORE_USE_SPDLOG ON)
  else()
    message(
      WARNING "#################################################################################\n"
              "spdlog was not found in your system. Re-run cmake configuration with option "
              "\"-DKALYPSSO_CORE_BUILD_SPDLOG=ON\" to enable building spdlog from source using  "
              "git submodule.\n"
              "#################################################################################")
  endif()

  set(USE_SPDLOG_BUILTIN FALSE)

endif(KALYPSSO_CORE_BUILD_SPDLOG)

# default log level in kalypsso-core everything below this value will be removed at compile time
# from the code here we chose the lowest level : everything is kept we can increase this level in a
# different build.
set(KALYPSSO_CORE_LOG_LEVELS SPDLOG_LEVEL_TRACE SPDLOG_LEVEL_DEBUG SPDLOG_LEVEL_INFO
                             SPDLOG_LEVEL_WARN SPDLOG_LEVEL_ERROR SPDLOG_LEVEL_CRITICAL)
set(KALYPSSO_CORE_LOG_LEVEL
    SPDLOG_LEVEL_TRACE
    CACHE STRING "Default value for SPDLOG_ACTIVE_LEVEL in spdlog macros")
set_property(CACHE KALYPSSO_CORE_LOG_LEVEL PROPERTY STRINGS ${KALYPSSO_CORE_LOG_LEVELS})
