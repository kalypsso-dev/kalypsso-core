add_executable(simple_2d "")
target_sources(simple_2d PUBLIC simple_2d.cpp)
target_link_libraries(simple_2d kalypsso::p4est_wrap kalypsso::config kalypsso::core_config)

# set(TEST_P4EST_CONFIGS unit three evil evil3 pillow moebius star cubed disk periodic rotwrap
# disk2d shell2d)
set(TEST_P4EST_CONFIGS unit shell2d)

foreach(test_p4est_config IN LISTS TEST_P4EST_CONFIGS)
  add_test(NAME p4est_simple_${test_p4est_config}
           COMMAND ${TEST_MPI_PREFIX_CMD} simple_2d ${test_p4est_config}
                   ${KALYPSSO_TEST_AMR_MAX_LEVEL})
  set_property(TEST p4est_simple_${test_p4est_config} PROPERTY LABELS p4est 2D)

  if(Kokkos_ENABLE_OPENMP)
    set_property(
      TEST p4est_simple_${test_p4est_config}
      PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS} OMP_PROC_BIND=spread
               OMP_PLACES=threads)
  endif()

endforeach()

add_executable(simple_3d "")
target_sources(simple_3d PUBLIC simple_3d.cpp)
target_link_libraries(simple_3d kalypsso::p4est_wrap kalypsso::config kalypsso::core_config)

# set(TEST_P8EST_CONFIGS unit periodic rotwrap twocubes twowrap rotcubes shell sphere)
set(TEST_P8EST_CONFIGS unit sphere)

foreach(test_p8est_config IN LISTS TEST_P8EST_CONFIGS)
  add_test(NAME p8est_simple_${test_p8est_config}
           COMMAND ${TEST_MPI_PREFIX_CMD} simple_3d ${test_p8est_config}
                   ${KALYPSSO_TEST_AMR_MAX_LEVEL})
  set_property(TEST p8est_simple_${test_p8est_config} PROPERTY LABELS p8est 3D)

  if(Kokkos_ENABLE_OPENMP)
    set_property(
      TEST p8est_simple_${test_p8est_config}
      PROPERTY ENVIRONMENT OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS} OMP_PROC_BIND=spread
               OMP_PLACES=threads)
  endif()

endforeach()
