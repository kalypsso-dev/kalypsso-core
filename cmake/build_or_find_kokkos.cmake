# Two alternatives:
#
# 1. If KALYPSSO_CORE_KOKKOS_BUILD is ON, we download kokkos sources and build them using FetchContent (which
#    actually uses add_subdirectory)
# 2. If KALYPSSO_CORE_KOKKOS_BUILD is OFF (default), we don't build kokkos, but use find_package for setup
#    (you must have kokkos already installed)

# NOTE about required C++ standard we better chose to set the minimum C++ standard level if not
# already done:
#
# * when building kokkos <  5.0.0, it defaults to c++-17
# * when building kokkos >= 5.0.0, it defaults to c++-20
# * when using installed kokkos, we set C++ standard according to kokkos version

#
# Does kalypsso-core builds kokkos (https://github.com/kokkos/kokkos) ?
#
option(KALYPSSO_CORE_KOKKOS_BUILD "Turn ON if you want to build kokkos (default: OFF)" OFF)

#
# Option to use git (instead of tarball release) for downloading kokkos
#
option(KALYPSSO_CORE_KOKKOS_USE_GIT
       "Turn ON if you want to use git to download Kokkos sources (default: OFF)" OFF)

#
# Options to specify target device backend
#

# set default backend
set(KALYPSSO_CORE_KOKKOS_BACKEND
    "Undefined"
    CACHE STRING "Kokkos default backend device")

# Set the possible values for kokkos backend device
set_property(CACHE KALYPSSO_CORE_KOKKOS_BACKEND PROPERTY STRINGS "OpenMP" "Cuda" "HIP" "Undefined")

# raise the minimum C++ standard level if not already done
# when build kokkos, it defaults to c++-20
# when using installed kokkos, it is not set, so defaulting to c++-20
# kokkos 5.0.0 requires c++-20 anyway
if (NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 20)
endif()

# check if user requested a build of kokkos use carefully, it may strongly increase build time
if(KALYPSSO_CORE_KOKKOS_BUILD)

  message("[kalypsso-core / kokkos] Building kokkos from source")

  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external)

  # Kokkos default build options

  # set install path
  list(APPEND KALYPSSO_CORE_KOKKOS_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR})

  # use predefined cmake args can be override on the command line
  if(KALYPSSO_CORE_KOKKOS_BACKEND MATCHES "Cuda")

    if((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_CUDA) OR (NOT Kokkos_ENABLE_CUDA))
      set(Kokkos_ENABLE_CUDA
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_CUDA_CONSTEXPR) OR (NOT Kokkos_ENABLE_CUDA_CONSTEXPR))
      set(Kokkos_ENABLE_CUDA_CONSTEXPR
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
       OR (NOT Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE))
      set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_CUDA_UVM) OR (NOT Kokkos_ENABLE_CUDA_UVM))
      set(Kokkos_ENABLE_CUDA_UVM
          OFF
          CACHE BOOL "")
    endif()

    # Note : cuda architecture will probed by kokkos cmake configure

  elseif(KALYPSSO_CORE_KOKKOS_BACKEND MATCHES "HIP")

    message(FATAL_ERROR "[kalypsso-core / kokkos] HIP backend to supported yet")

    if(NOT KALYPSSO_CORE_ENABLE_GPU_HIP)
      message(
        FATAL_ERROR
          "[kalypsso-core / kokkos] You can't use Kokkos::HIP backend if KALYPSSO_CORE_ENABLE_GPU_HIP is OFF"
      )
    endif()

    if((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_HIP) OR (NOT Kokkos_ENABLE_HIP))
      set(Kokkos_ENABLE_HIP
          ON
          CACHE BOOL "")
    endif()

  elseif(KALYPSSO_CORE_KOKKOS_BACKEND MATCHES "OpenMP")

    if((NOT DEFINED Kokkos_ENABLE_HWLOC) OR (NOT Kokkos_ENABLE_HWLOC))
      set(Kokkos_ENABLE_HWLOC
          ON
          CACHE BOOL "")
    endif()

    if((NOT DEFINED Kokkos_ENABLE_OPENMP) OR (NOT Kokkos_ENABLE_OPENMP))
      set(Kokkos_ENABLE_OPENMP
          ON
          CACHE BOOL "")
    endif()

  elseif(KALYPSSO_CORE_KOKKOS_BACKEND MATCHES "Undefined")

    message(
      FATAL_ERROR "[kalypsso-core / kokkos] You must chose a valid KALYPSSO_CORE_KOKKOS_BACKEND !")

  endif()

  # find_package(Git REQUIRED)
  include(FetchContent)

  if(KALYPSSO_CORE_KOKKOS_USE_GIT)
    message("[kalypsso-core] Building kokkos from source using git sources")
    FetchContent_Declare(
      kokkos_external
      SYSTEM
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 5.1.0)
  else()
    message("[kalypsso-core] Building kokkos from source using git submodule")
    FetchContent_Declare(
      kokkos_external
      SYSTEM
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/kokkos)
  endif()

  # Import kokkos targets (download, and call add_subdirectory)
  FetchContent_MakeAvailable(kokkos_external)

  if(TARGET Kokkos::kokkos)
    message("[kalypsso-core / kokkos] Kokkos found (using FetchContent)")
    set(KALYPSSO_CORE_KOKKOS_FOUND True)
    set(HAVE_KOKKOS 1)
  else()
    message(
      "[kalypsso-core / kokkos] we shouldn't be here. We've just integrated kokkos build into kalypsso-core build !"
    )
  endif()

  set(KALYPSSO_CORE_KOKKOS_BUILTIN TRUE)

else()

  #
  # check if an already installed kokkos exists
  #
  find_package(Kokkos 5.1.0 CONFIG REQUIRED)

  if(TARGET Kokkos::kokkos)

    kokkos_check(DEVICES "OpenMP")

    if(KALYPSSO_CORE_ENABLE_GPU_CUDA)
      # kokkos_check is defined in KokkosConfigCommon.cmake
      kokkos_check(DEVICES "Cuda")
      kokkos_check(OPTIONS CUDA_CONSTEXPR)
    elseif(KALYPSSO_CORE_ENABLE_GPU_HIP)
      # TODO
      kokkos_check(DEVICES "HIP")
    endif()

    message("[kalypsso-core / kokkos] Kokkos found via find_package")
    set(KALYPSSO_CORE_KOKKOS_FOUND True)
    set(HAVE_KOKKOS 1)

  else()

    message(
      FATAL_ERROR
        "[kalypsso-core / kokkos] Kokkos is required but not found by find_package. Please adjust your env variable CMAKE_PREFIX_PATH (or Kokkos_ROOT) to where Kokkos is installed on your machine !"
    )

  endif()

endif()
