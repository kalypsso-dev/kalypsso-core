set(KALYPSSO_TEST_AMR_MAX_LEVEL
    3
    CACHE STRING "Number of maximum level of AMR for tests")

if(KALYPSSO_CORE_USE_MPI)
  set(KALYPSSO_TEST_MPIEXEC_NUMPROCS
      2
      CACHE STRING "Number of MPI process for tests")
  set(TEST_MPI_PREFIX_CMD ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                          ${KALYPSSO_TEST_MPIEXEC_NUMPROCS})
endif()

if(Kokkos_ENABLE_OPENMP)
  set(KALYPSSO_TEST_OMP_NUM_THREADS
      2
      CACHE STRING "Number of OpenMP threads for tests")
endif()

add_subdirectory(test_common)

add_subdirectory(kokkos)

add_subdirectory(p4est_simple)

add_subdirectory(p4est_wrapper)

add_subdirectory(io)

add_subdirectory(AMRmesh)

add_subdirectory(AMRkey)

add_subdirectory(MeshMap)

if(KALYPSSO_CORE_USE_SPDLOG)
  add_subdirectory(spdlog)
endif(KALYPSSO_CORE_USE_SPDLOG)

add_subdirectory(monitoring)

add_subdirectory(HighFive)

add_subdirectory(container)

add_subdirectory(core)

if(KALYPSSO_CORE_USE_CPPTRACE)
  add_subdirectory(cpptrace)
endif(KALYPSSO_CORE_USE_CPPTRACE)

add_subdirectory(utils)
