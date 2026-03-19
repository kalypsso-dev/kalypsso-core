#
# Kokkos tests / examples
#

add_executable(query_device_kokkos query_device_kokkos.cpp)
target_link_libraries(query_device_kokkos PUBLIC Kokkos::kokkos kalypsso::core_config)
if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(query_device_kokkos PUBLIC MPI::MPI_CXX)
endif(KALYPSSO_CORE_USE_MPI)
add_test(NAME query_device_kokkos COMMAND query_device_kokkos)
set_property(TEST query_device_kokkos PROPERTY LABELS kokkos)
if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST query_device_kokkos PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                      OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

add_executable(test_kokkos_teamvector test_kokkos_teamvector.cpp)
target_link_libraries(test_kokkos_teamvector PUBLIC kalypsso::core Kokkos::kokkos)
if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(test_kokkos_teamvector PUBLIC kalypsso::mpiUtils MPI::MPI_CXX)
endif(KALYPSSO_CORE_USE_MPI)
add_test(NAME kokkos_teamvector COMMAND test_kokkos_teamvector)
set_property(TEST kokkos_teamvector PROPERTY LABELS kokkos)
if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST kokkos_teamvector PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                    OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

add_executable(test_kokkos_mdrange test_kokkos_mdrange.cpp)
target_link_libraries(test_kokkos_mdrange PUBLIC kalypsso::core Kokkos::kokkos)
if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(test_kokkos_mdrange PUBLIC kalypsso::mpiUtils MPI::MPI_CXX)
endif(KALYPSSO_CORE_USE_MPI)
add_test(NAME kokkos_mdrange COMMAND test_kokkos_mdrange)
set_property(TEST kokkos_mdrange PROPERTY LABELS kokkos)
if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST kokkos_mdrange PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                 OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

add_executable(test_kokkos_team_mdrange test_kokkos_team_mdrange.cpp)
target_link_libraries(test_kokkos_team_mdrange PUBLIC kalypsso::core Kokkos::kokkos)
if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(test_kokkos_team_mdrange PUBLIC kalypsso::mpiUtils MPI::MPI_CXX)
endif(KALYPSSO_CORE_USE_MPI)
add_test(NAME kokkos_team_mdrange COMMAND test_kokkos_team_mdrange)
set_property(TEST kokkos_team_mdrange PROPERTY LABELS kokkos)
if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST kokkos_team_mdrange PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                      OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# ##################################################################################################
add_executable(test_unordered_map test_unordered_map.cpp)
target_link_libraries(test_unordered_map PUBLIC Kokkos::kokkos kalypsso::core_config)
add_test(NAME unordered_map COMMAND test_unordered_map)
set_property(TEST unordered_map PROPERTY LABELS kokkos)
if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST unordered_map PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# ##################################################################################################
add_executable(test_morton_key test_morton_key.cpp)
target_link_libraries(test_morton_key PUBLIC Kokkos::kokkos kalypsso::core_config)
add_test(NAME morton_key COMMAND test_morton_key)
set_property(TEST morton_key PROPERTY LABELS kokkos)
if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST morton_key PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                             OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# ##################################################################################################
if(KALYPSSO_CORE_USE_DOUBLE)
  add_executable(compute_mandelbrot_amr compute_mandelbrot_amr.cpp)
  target_link_libraries(compute_mandelbrot_amr PUBLIC Kokkos::kokkos kalypsso::io kalypsso::core
                                                      kalypsso::core_config)

  # cmake-format: off
# if(KALYPSSO_CORE_USE_MPI)
#   target_link_libraries(compute_mandelbrot_amr PUBLIC
#                         kalypsso::mpiUtils)
# endif(KALYPSSO_CORE_USE_MPI)
# add_test(NAME compute_mandelbrot_amr
#          COMMAND compute_mandelbrot_amr)
# set_property(TEST compute_mandelbrot_amr
#              PROPERTY LABELS kokkos)
# if(Kokkos_ENABLE_OPENMP)
#   set_property(TEST compute_mandelbrot_amr PROPERTY ENVIRONMENT
#                OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS} OMP_PROC_BIND=spread
#                OMP_PLACES=threads)
# endif()
# cmake-format: on
endif()

# ##################################################################################################
add_executable(test_parallel_scan test_parallel_scan.cpp)
target_link_libraries(test_parallel_scan PUBLIC Kokkos::kokkos kalypsso::core)

if(KALYPSSO_CORE_USE_MPI)
  target_link_libraries(test_parallel_scan PUBLIC kalypsso::mpiUtils)
endif(KALYPSSO_CORE_USE_MPI)

add_test(NAME parallel_scan COMMAND test_parallel_scan)
set_property(TEST parallel_scan PROPERTY LABELS kokkos)
if(Kokkos_ENABLE_OPENMP)
  set_property(
    TEST parallel_scan PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS}
                                OMP_PROC_BIND=spread OMP_PLACES=threads)
endif()

# ##################################################################################################
if(KALYPSSO_CORE_USE_MPI)
  add_executable(test_kokkos_mpi test_kokkos_mpi.cpp)
  target_link_libraries(test_kokkos_mpi PUBLIC Kokkos::kokkos kalypsso::core kalypsso::mpiUtils)
endif(KALYPSSO_CORE_USE_MPI)
