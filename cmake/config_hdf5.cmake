# ##################################################################################################
# HDF5
# ##################################################################################################
# prefer using parallel HDF5 when build with mpi
if(KALYPSSO_CORE_USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(KALYPSSO_CORE_USE_MPI)

if(KALYPSSO_CORE_USE_HDF5)
  find_package(
    HDF5
    COMPONENTS C
    REQUIRED)
  if(HDF5_FOUND)
    set(KALYPSSO_CORE_USE_HDF5_VERSION ${HDF5_VERSION})
    include_directories(${HDF5_INCLUDE_DIRS})
    set(MY_HDF5_LIBS hdf5 hdf5_cpp)
    if(HDF5_IS_PARALLEL)
      set(KALYPSSO_CORE_USE_HDF5_PARALLEL True)
      message(STATUS "PARALLEL HDF5 found, version=${KALYPSSO_CORE_USE_HDF5_VERSION}")
    else()
      message(STATUS "SERIAL HDF5 found, version=${KALYPSSO_CORE_USE_HDF5_VERSION}")
    endif()
  else()
    message(FATAL_ERROR "HDF5 requested but not found.")
  endif(HDF5_FOUND)
endif(KALYPSSO_CORE_USE_HDF5)
