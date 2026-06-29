#
# Declare optional features
#

option(KALYPSSO_CORE_USE_MPI "Activate / want MPI build" ON)
option(KALYPSSO_CORE_USE_VTK "Activate / want VTK build" OFF)
option(KALYPSSO_CORE_USE_DOUBLE "build with double precision" ON)
option(KALYPSSO_CORE_USE_HDF5 "build HDF5 input/output support" ON)
option(KALYPSSO_CORE_USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option(KALYPSSO_CORE_USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option(KALYPSSO_CORE_USE_MPI_CUDA_AWARE_ENFORCED
       "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)
option(KALYPSSO_CORE_TIMING_ENABLED "build with run-timing measurements enabled (default ON)" ON)
option(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLE "build with nvtx annotations enabled (default OFF)" OFF)
option(KALYPSSO_CORE_BUILD_PARAVIEW_PLUGIN
       "build paraview plugin for loading hdf5 data into a vtkNonOverlappingAMR object." OFF)
option(KALYPSSO_CORE_DEBUG_BOUNDS_CHECK
       "Check all array access bounds at runtime (for debug purpose)." OFF)
option(KALYPSSO_CORE_ENABLE_WARNINGS "Enable developer warnings (default ON)" ON)
option(KALYPSSO_CORE_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors (default OFF)."
       OFF)
option(KALYPSSO_CORE_DISABLE_DEPRECATED_WARNINGS
       "Disable deprecated declaration warnings (default OFF)" OFF)
option(KALYPSSO_CORE_USE_CPPTRACE "Use cpptrace to print a stack trace upon failure (default OFF)."
       OFF)
option(KALYPSSO_CORE_CONFIGMAP_VERBOSE
       "Make ConfigMap verbose, i.e. log if a variable is set to default value (default OFF).")

# the following option are only active when Kokkos::Cuda backend is selected, they are useful for
# Nvidia profiling tools NCU
option(KALYPSSO_CORE_CUDA_LINEINFO
       "When Cuda backend is enabled, add compile flag \"-lineinfo\" (default OFF)" OFF)
option(KALYPSSO_CORE_CUDA_RES_USAGE
       "When Cuda backend is enabled, add compile flag \"-res_usage\" (default OFF)" OFF)
option(KALYPSSO_CORE_CUDA_PTXAS_VERBOSE
       "When Cuda backend is enabled, add compile flag \"--ptxas-options -v\" (default OFF)" OFF)

# option (USE_MOOD "build MOOD numerical schemes" OFF)

# option (USE_SDM "build Spectral Difference Method numerical schemes" OFF)

#
# documentation related options
#
option(KALYPSSO_CORE_BUILD_DOC "Enable / disable documentation build" OFF)

# documentation type - the only valid values are : doxygen and mkdocs
if(NOT KALYPSSO_CORE_DOC)
  set(KALYPSSO_CORE_DOC
      "doxygen"
      CACHE STRING "documentation type (doxygen or mkdocs)" FORCE)
  set_property(CACHE KALYPSSO_CORE_DOC PROPERTY STRINGS "doxygen" "mkdocs")
endif()

option(KALYPSSO_CORE_ENABLE_UNIT_TESTING "Enable unit testing" OFF)

if(${KALYPSSO_CORE_KOKKOS_BACKEND} STREQUAL "Cuda" OR ${KALYPSSO_CORE_KOKKOS_BACKEND} STREQUAL
                                                      "HIP")
  set(KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE ON)
else()
  set(KALYPSSO_CORE_INSTANTIATE_HOST_TEMPLATE OFF)
endif()

# use clang-tidy default : OFF but we provide a default config
option(KALYPSSO_CORE_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)

if(KALYPSSO_CORE_ENABLE_CLANG_TIDY)

  # BEWARE : kokkos checks are only available when using a special version of clang-tidy see
  # https://github.com/kokkos/kokkos-tutorials/blob/main/LectureSeries/KokkosTutorial_07_Tools.pdf
  # you will need to re-build clang-tidy from https://github.com/kokkos/llvm-project using branch
  # kokkos-ensure-kokkos-function
  set(DEFAULT_CLANG_TIDY_CONFIG "clang-tidy;-checks=kokkos-*")

  if(NOT CMAKE_CXX_CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY "${DEFAULT_CLANG_TIDY_CONFIG}")
  endif()

endif(KALYPSSO_CORE_ENABLE_CLANG_TIDY)
