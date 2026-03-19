# set minimal version to the one requested by kokkos
cmake_minimum_required(VERSION 3.18)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
  message(STATUS "Setting policy CMP0076 to use new behavior")
  cmake_policy(SET CMP0076 NEW)
endif()

# CMake 3.24 and above prefers to set the timestamps of all extracted contents to the time of the
# extraction.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  cmake_policy(SET CMP0135 NEW)
endif()

set(KALYPSSO_CORE_VERSION "0.0.0")

# deduce KALYPSSO_CORE_SHORT_VERSION using regex
string(REGEX MATCH "^[0-9]+\.[0-9]+\.[0-9]+" KALYPSSO_CORE_SHORT_VERSION ${KALYPSSO_CORE_VERSION})
if("${KALYPSSO_CORE_SHORT_VERSION}" STREQUAL "")
  message(
    FATAL_ERROR
      "Unable to compute short version from KALYPSSO_CORE_VERSION=${KALYPSSO_CORE_VERSION}")
endif()

project(
  kalypsso-core
  VERSION ${KALYPSSO_CORE_SHORT_VERSION}
  LANGUAGES C CXX Fortran)

# project_name in small caps
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_SC)

#
# default local cmake macro repository
#
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#
# Prevent from build in source tree
#
include(cmake/prevent_build_in_source.cmake)
prevent_build_in_source()

#
# Init build type: Release, Debug, ... (default is Release)
#
include(cmake/initBuildType.cmake)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
# kalypsso options
#
include(cmake/options.cmake)

# C++17 is required for Kokkos >= 4.0 CMAKE_CXX_STANDARD is the default value for the CXX_STANDARD
# target property
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_EXTENSIONS OFF)
else()
  message(STATUS "CMAKE_CXX_STANDARD already set to ${CMAKE_CXX_STANDARD}")
endif()

#
# kalypsso developer settings
#

add_library(kalypsso_warnings INTERFACE)

include(cmake/CompilerWarnings.cmake)

if(KALYPSSO_CORE_ENABLE_WARNINGS)
  kalypsso_set_project_warnings(
    kalypsso_warnings
    ${KALYPSSO_CORE_ENABLE_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    ""
    "")
else()
  kalypsso_set_project_warnings(
    kalypsso_warnings
    ${KALYPSSO_CORE_ENABLE_WARNINGS_AS_ERRORS}
    CLANG_C_WARNINGS_DEFAULT
    CLANG_CXX_WARNINGS_DEFAULT
    GCC_C_WARNINGS_DEFAULT
    GCC_CXX_WARNINGS_DEFAULT
    CUDA_WARNINGS_DEFAULT)
endif()
add_library(kalypsso::warnings ALIAS kalypsso_warnings)

include(cmake/setup_install.cmake)
install(
  TARGETS kalypsso_warnings
  EXPORT kalypsso-core-targets
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#
# Build or find p4est ?
#
include(cmake/build_or_find_p4est.cmake)

#
# Build or find spdlog ?
#
include(cmake/build_or_find_spdlog.cmake)

#
# Build or find cnpy ?
#
include(cmake/build_or_find_cnpy.cmake)

#
# Build or find nvtx3 ?
#
include(cmake/build_or_find_nvtx3.cmake)

#
# Configure better-enums
#
include(cmake/better-enums-config.cmake)

#
# Configure cpptrace
#
include(cmake/build_or_find_cpptrace.cmake)

#
# check MPI, VTK, HDF5, PNETCDF, ...
#
include(cmake/config_mpi.cmake)
include(cmake/config_vtk.cmake)
include(cmake/config_hdf5.cmake)
include(cmake/config_pnetcdf.cmake)
include(cmake/config_highfive.cmake)

#
# Generate kalypsso_core_config.h and kalypsso_core_version.h (with git info and build date)
#
include(cmake/generate_config_h.cmake)

# ##################################################################################################
# ##################################################################################################
# now we build kalypsso
# ##################################################################################################
# ##################################################################################################

# build kokkos
include(cmake/build_or_find_kokkos.cmake)

# build kalypsso
add_subdirectory(src)

# build additional tools
add_subdirectory(tools)

include(CTest)

add_subdirectory(test)

# add target kalypsso-test
if(KALYPSSO_CORE_ENABLE_UNIT_TESTING)
  add_subdirectory(unit_test)
endif(KALYPSSO_CORE_ENABLE_UNIT_TESTING)

# add target kalypsso-doc
if(KALYPSSO_CORE_BUILD_DOC)
  add_subdirectory(doc)
endif(KALYPSSO_CORE_BUILD_DOC)

# ##################################################################################################
# Install ####
# ##################################################################################################
include(cmake/setup_install.cmake)

# make a list of targets to installed (kalypsso itself and all dependencies that are built along)
set(TARGETS_LIST
    kalypsso_shared
    kalypsso_config
    kalypsso_log
    kalypsso_monitoring
    kalypsso_mpiUtils
    kalypsso_p4est_wrap
    HighFive
    HighFiveInclude)
if(KALYPSSO_CORE_KOKKOS_BUILD)
  list(
    APPEND
    targets_list
    kokkos
    kokkoscore
    kokkoscontainers
    kokkosalgorithms
    kokkossimd
    LIBDL)
  if(Kokkos_ENABLE_CUDA)
    list(APPEND targets_list CUDA)
  endif()

  # TODO update targets_list when other Kokkos backend are used
endif(KALYPSSO_CORE_KOKKOS_BUILD)

# Use a namespace because CMake provides better diagnostics for namespaced imported targets.
export(
  TARGETS ${targets_list}
  NAMESPACE ${PROJECT_NAME_SC}::
  FILE ${PROJECT_BINARY_DIR}/kalypsso-core-targets.cmake)

install(
  EXPORT kalypsso-core-targets
  NAMESPACE kalypsso::
  FILE kalypsso-core-targets.cmake
  DESTINATION ${INSTALL_CMAKE_DIR})

#
# Install cmake related files (targets, exports, ...)
#

set(VERSION_CONFIG ${PROJECT_BINARY_DIR}/${PROJECT_NAME_SC}-config-version.cmake)
set(PROJECT_CONFIG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME_SC}-config.cmake.in)
set(PROJECT_CONFIG_DST ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME_SC}-config.cmake)

install(FILES ${PROJECT_CONFIG_DST} ${VERSION_CONFIG} DESTINATION ${INSTALL_CMAKE_DIR})

include(CMakePackageConfigHelpers)

# Generate the version, config and target files into the build directory.
write_basic_package_version_file(
  ${VERSION_CONFIG}
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

# generate cmake my_package-config.cmake file
configure_package_config_file(${PROJECT_CONFIG_SRC} ${PROJECT_CONFIG_DST}
                              INSTALL_DESTINATION ${INSTALL_CMAKE_DIR})

#
# Install third-party libraries
#

# install better-enums (header only)
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/external/better-enums
  DESTINATION ${INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "test" EXCLUDE
  PATTERN "doc" EXCLUDE
  PATTERN "example" EXCLUDE
  PATTERN "script" EXCLUDE
  PATTERN ".github" EXCLUDE)

# #################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("")
message("  CMake version          : ${CMAKE_VERSION}")
if(NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CMake install prefix   : ${CMAKE_INSTALL_PREFIX}")
message("  CMake system processor : ${CMAKE_SYSTEM_PROCESSOR}")
message("  CMake system name (OS) : ${CMAKE_SYSTEM_NAME}")
message("")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(DEFINED CMAKE_CXX_COMPILER_WRAPPER)
  message("                 ${CMAKE_CXX_COMPILER_WRAPPER}")
endif()
message("                 ${CMAKE_CXX_COMPILER}")
message("  CMAKE_CXX_STANDARD : ${CMAKE_CXX_STANDARD}")
message("")

if(KALYPSSO_CORE_USE_MPI)
  message("  MPI enabled")
else(KALYPSSO_CORE_USE_MPI)
  message("  MPI not enabled")
endif(KALYPSSO_CORE_USE_MPI)

message("  Kokkos version               = ${Kokkos_VERSION}")
message("  Kokkos_CXX_COMPILER          = ${Kokkos_CXX_COMPILER}")
message("  Kokkos_CXX_COMPILER_ID       = ${Kokkos_CXX_COMPILER_ID}")
message("  Kokkos_CXX_STANDARD          = ${Kokkos_CXX_STANDARD}")
message("  Kokkos_OPTIONS               = ${Kokkos_OPTIONS}")
message("  Kokkos_TPLS                  = ${Kokkos_TPLS}")
message("  Kokkos_DIRS                  = ${Kokkos_DIR}")

message("  KALYPSSO_CORE_KOKKOS_BACKEND = ${KALYPSSO_CORE_KOKKOS_BACKEND}")

if(Kokkos_ENABLE_OPENMP)
  message("  Kokkos_ENABLE_OPENMP         = ${Kokkos_ENABLE_OPENMP}")
endif()

if(Kokkos_ENABLE_CUDA)
  message("  Kokkos_ENABLE_CUDA           = ${Kokkos_ENABLE_CUDA}")

  if((${Kokkos_CUDA_LAMBDA_ENABLED}) OR (${Kokkos_ENABLE_CUDA_LAMBDA}))
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_LAMBDA    = OFF")
  endif()

  if((${Kokkos_CUDA_CONSTEXPR_ENABLED}) OR (${Kokkos_ENABLE_CUDA_CONSTEXPR}))
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_CONSTEXPR = OFF")
  endif()

  if((${Kokkos_CUDA_UVM_ENABLED}) OR (${Kokkos_ENABLE_CUDA_UVM}))
    message("  Kokkos_ENABLE_CUDA_UVM       = ON")
  else()
    message("  Kokkos_ENABLE_CUDA_UVM       = OFF")
  endif()

  message("  Kokkos CUDA   flags          = ${KOKKOS_CUDA_OPTIONS}")
endif(Kokkos_ENABLE_CUDA)

if(Kokkos_ENABLE_HIP)
  message("  Kokkos_ENABLE_HIP            = ${Kokkos_ENABLE_HIP}")
endif(Kokkos_ENABLE_HIP)

if((${Kokkos_TPLS_HWLOC_ENABLED}) OR (${Kokkos_ENABLE_HWLOC}))
  message("  Kokkos_ENABLE_HWLOC          = ON")
else()
  message("  Kokkos_ENABLE_HWLOC          = OFF")
endif()

message("  Kokkos architecture          = ${Kokkos_ARCH}")

message("")
if(KALYPSSO_CORE_BUILD_P4EST)
  message("  P4EST built using ExternalProject_Add")
else()
  message("  P4EST found in environment")
endif()

message("")
if(KALYPSSO_CORE_BUILD_SPDLOG)
  message("  SPDLOG built using ExternalProject_Add")
else()
  message("  SPDLOG found in environment")
endif()

message("")
if(KALYPSSO_CORE_BUILD_CNPY)
  message("  CNPY built using ExternalProject_Add")
else()
  message("  CNPY found in environment")
endif()

if(KALYPSSO_CORE_USE_HDF5)
  if(HDF5_FOUND)
    message("  HDF5 found version    : ${HDF5_VERSION}")
    message("  HDF5 definitions      : ${HDF5_DEFINITIONS}")
    message("  HDF5 parallel         : ${HDF5_IS_PARALLEL}")
    message("  HDF5 includes dirs    : ${HDF5_INCLUDE_DIRS}")
    message("  HDF5 libraries        : ${HDF5_LIBRARIES}")
  endif(HDF5_FOUND)
endif(KALYPSSO_CORE_USE_HDF5)
message("")

if(KALYPSSO_CORE_USE_SPDLOG)
  message("  SPDLOG                : enabled")
else(KALYPSSO_CORE_USE_SPDLOG)
  message("  SPDLOG                : disabled")
endif(KALYPSSO_CORE_USE_SPDLOG)
message("      builtin                : ${USE_SPDLOG_BUILTIN}")
message("      version                : ${spdlog_VERSION}")

if(KALYPSSO_CORE_USE_CNPY)
  message("  CNPY                  : enabled")
else(KALYPSSO_CORE_USE_CNPY)
  message("  CNPY                  : disabled")
endif(KALYPSSO_CORE_USE_CNPY)

if(KALYPSSO_CORE_USE_NVTX3)
  message("  NVTX                  : enabled")
else(KALYPSSO_CORE_USE_NVTX3)
  message("  NVTX                  : disabled")
endif(KALYPSSO_CORE_USE_NVTX3)

if(KALYPSSO_CORE_BUILD_DOC)
  message("   Building doc of type ${KALYPSSO_CORE_DOC}")
endif(KALYPSSO_CORE_BUILD_DOC)
message("")
