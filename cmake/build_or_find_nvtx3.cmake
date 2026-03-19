#
# Nvtx can be used even if Kokkos::Cuda backend is not activated, for a pure CPU run. All we need to
# perform tracing is to run on a platform where cuda driver and nsys profile/tracing tools are
# installed
#

#
# Nvtx is already shipped with nvcc toolkit and nvhpc compiler, but we want to be able to instrument
# code event for pure CPU build and run; so we need a way to provide nvtx when not building with
# nvcc or nvc++
#

# 1. if using an Nvidia compiler, we don't need to download nvtx sources (they are already available)
# 2. else we need to download nvtx sources to enable NVTX annotations

#
# Option to use git (instead of tarball release) for downloading nvtx
#
option(
  KALYPSSO_CORE_NVTX_USE_GIT
  "Turn ON if you want to use git (instead of archive file) to download nvtx sources (default: ON)"
  ON)

set(KALYPSSO_CORE_USE_NVTX_FROM_GITHUB FALSE)
set(KALYPSSO_CORE_USE_NVTX3 OFF)

# If use requested NVTX annotations, we need to check which compiler is used. If compiler is not an
# nvidia compiler, we need to download nvtx sources
if(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLE)

  # check if compiler is nvcc or nvc++, nvtx is available
  if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC" OR KOKKOS_CXX_COMPILER_ID MATCHES "NVIDIA")
    message("[kalypsso-core / nvtx] NVTX found, using a nvidia compiler")
    set(KALYPSSO_CORE_NVTX3_FOUND TRUE)
  else()

    message("[kalypsso-core / nvtx] Downloading nvtx sources")

    set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external)

    include(FetchContent)

    if(KALYPSSO_CORE_NVTX_USE_GIT)
      FetchContent_Declare(
        nvtx_external
        SYSTEM
        GIT_REPOSITORY https://github.com/nvidia/nvtx.git
        GIT_TAG release-v3
        SOURCE_SUBDIR c)
    else()
      FetchContent_Declare(
        nvtx_external
        SYSTEM
        URL https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.1.0.tar.gz)
    endif()

    # Import nvtx targets (download, and call add_subdirectory)
    FetchContent_MakeAvailable(nvtx_external)

    if(TARGET nvtx3-cpp)
      message("[kalypsso-core / nvtx] NVTX found (using FetchContent)")
      set(KALYPSSO_CORE_NVTX3_FOUND True)
    else()
      message(
        "[kalypsso-core / nvtx] we shouldn't be here. We've just integrated nvtx build into kalypsso-core build !"
      )
    endif()

    set(KALYPSSO_CORE_USE_NVTX_FROM_GITHUB TRUE)

  endif()

endif(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLE)

if(KALYPSSO_CORE_NVTX3_FOUND)
  set(KALYPSSO_CORE_USE_NVTX3 ON)
endif()

# symbol KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED is used to feed kalypsso_core_config.h.cmake
if(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLE AND KALYPSSO_CORE_NVTX3_FOUND)
  set(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED TRUE)
else()
  set(KALYPSSO_CORE_NVTX_ANNOTATION_ENABLED FALSE)
endif()
